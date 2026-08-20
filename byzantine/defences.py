# ===========================================================================
# Project:      On the Byzantine-Resilience of Distillation-Based Federated Learning - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2402.12265
# File:         byzantine/defences.py
# Description:  Byzantine Defence classes
# ===========================================================================
import math
import sys

import torch
from tqdm.auto import tqdm

from utilities import Utilities as Utils


#### Defence Base Class
class NoDefence:
    """Defence base class"""

    def __init__(self, **kwargs):
        self.config = kwargs['config']
        self.clients = kwargs['clients']
        self.runner = kwargs['runner_instance']
        self._active_client_ids = None

    def set_active_client_ids(self, client_ids):
        """Associate the current upload list with stable client IDs.

        Probe filtering can remove uploads before aggregation. Stateful defences
        therefore must not assume that the list position is a permanent client ID.
        """
        self._active_client_ids = [int(client_id) for client_id in client_ids]

    def _get_active_client_ids(self, n_uploads):
        if self._active_client_ids is None:
            client_ids = [int(client.client_id) for client in self.clients]
        else:
            client_ids = self._active_client_ids
        if len(client_ids) != n_uploads:
            raise ValueError(
                "The number of active client IDs must match the number of uploaded logits."
            )
        return client_ids

    def _config_value(self, key, default=None):
        try:
            value = self.config[key]
        except (KeyError, TypeError):
            value = getattr(self.config, key, default)
        return default if value is None else value

    def get_aggregated_model(self, client_model_list: list):
        return Utils.average_client_models(client_model_list)

    def get_aggregated_predictions(self, client_predictions_list: list):
        avg_pred = Utils.average_client_predictions(client_predictions_list=client_predictions_list,
                                                output_type='softmax')
        pred_stack = torch.stack(client_predictions_list).permute(1,2,0)
        outlier_scores = torch.linalg.norm(pred_stack - avg_pred.unsqueeze(-1), dim=1)
        mean_outlier_scores = torch.mean(outlier_scores, dim=0)
        return avg_pred, mean_outlier_scores

    def get_aggregated_logits(self, client_logits_list: list):
        avg_logits = torch.mean(torch.stack(client_logits_list), dim=0)
        logits_stack = torch.stack(client_logits_list).permute(1, 2, 0)
        outlier_scores = torch.linalg.norm(logits_stack - avg_logits.unsqueeze(-1), dim=1)
        mean_outlier_scores = torch.mean(outlier_scores, dim=0)
        return avg_logits, mean_outlier_scores


class LogitAggregationDefence(NoDefence):
    """Base class for defences defined only for FedDistill logit uploads."""

    def get_aggregated_model(self, client_model_list: list):
        raise NotImplementedError(
            f"{self.__class__.__name__} aggregates client logits and is only supported "
            "with a distillation strategy that uploads logits."
        )

    @staticmethod
    def _mean_logit_distances(logits_stack, aggregate):
        return torch.linalg.vector_norm(
            (logits_stack - aggregate.unsqueeze(0)).reshape(logits_stack.size(0), -1),
            dim=1,
        )


class TriMean(LogitAggregationDefence):
    """Coordinate-wise trimmed mean over client logit uploads.

    For every public-sample/class coordinate, discard the lowest and highest
    ``floor(trim_ratio * n_uploads)`` client logits and average the remainder.
    """

    @torch.no_grad()
    def get_aggregated_logits(self, client_logits_list: list):
        logits_stack = torch.stack(client_logits_list)
        n_uploads = logits_stack.size(0)
        trim_ratio = float(self._config_value("trimean_trim_ratio", 0.2))
        if not 0.0 <= trim_ratio < 0.5:
            raise ValueError("trimean_trim_ratio must be in [0, 0.5).")

        trim_count = int(math.floor(trim_ratio * n_uploads))
        if 2 * trim_count >= n_uploads:
            raise ValueError(
                "TriMean would remove every uploaded logit; reduce trimean_trim_ratio "
                "or use more participating clients."
            )

        sorted_logits, _ = torch.sort(logits_stack, dim=0)
        if trim_count:
            aggregate = sorted_logits[trim_count:n_uploads - trim_count].mean(dim=0)
        else:
            aggregate = sorted_logits.mean(dim=0)
        outlier_scores = self._mean_logit_distances(logits_stack, aggregate)
        sys.stdout.write(
            f"[TriMean] Trimmed {trim_count} lowest and {trim_count} highest client logits "
            f"per coordinate (n={n_uploads}, ratio={trim_ratio:.4f}).\n"
        )
        return aggregate, outlier_scores


class Krum(LogitAggregationDefence):
    """Krum/Multi-Krum adapted to whole client logit matrices.

    Each client's complete ``[public_samples, classes]`` logit tensor is
    flattened into one update vector. Krum scores are sums of the nearest
    ``n - f - 2`` squared L2 distances. ``krum_num_selected=1`` is Krum;
    larger values implement Multi-Krum by averaging the selected uploads.

    When Probe core-sigma filtering is enabled, ``krum_residual_f`` is the
    pre-registered hard upper bound on Byzantine uploads that may remain.
    ``krum_f`` overrides this value for an explicitly configured experiment.
    """

    @torch.no_grad()
    def get_aggregated_logits(self, client_logits_list: list):
        logits_stack = torch.stack(client_logits_list)
        n_uploads = logits_stack.size(0)
        probe_filtered = bool(self._config_value("probe_core_sigma_filter", False))
        configured_f = self._config_value("krum_f", None)
        if configured_f is None:
            if probe_filtered:
                configured_f = self._config_value("krum_residual_f", 3)
                f_source = "residual upper bound"
            else:
                configured_f = self._config_value("n_byzantine_clients", 0)
                f_source = "configured Byzantine upper bound"
        else:
            f_source = "explicit krum_f"
        f = int(configured_f)
        if f < 0:
            raise ValueError("krum_f must be non-negative.")
        if n_uploads <= 2 * f + 2:
            fallback = str(self._config_value("krum_fallback", "mean") or "error").lower()
            if probe_filtered and fallback == "mean":
                aggregate = logits_stack.mean(dim=0)
                outlier_scores = self._mean_logit_distances(logits_stack, aggregate)
                sys.stdout.write(
                    "[Krum] Abstained because the residual upper bound is infeasible "
                    f"(n={n_uploads}, f={f}); falling back to mean over SPC-retained uploads.\n"
                )
                return aggregate, outlier_scores
            raise ValueError(
                f"Krum requires n > 2f + 2, but received n={n_uploads}, f={f}. "
                "After Probe filtering, reduce --krum_residual_f only when that reduced "
                "residual bound is an explicit security assumption."
            )

        neighbor_count = n_uploads - f - 2
        selected_count = int(self._config_value("krum_num_selected", 1))
        if not 1 <= selected_count <= neighbor_count:
            raise ValueError(
                f"krum_num_selected must be in [1, {neighbor_count}] for n={n_uploads}, f={f}."
            )

        flat_logits = logits_stack.reshape(n_uploads, -1).float()
        pairwise_distances = torch.cdist(flat_logits, flat_logits, p=2).square()
        pairwise_distances.fill_diagonal_(float("inf"))
        krum_scores = torch.topk(
            pairwise_distances, k=neighbor_count, largest=False, dim=1
        ).values.sum(dim=1)
        selected_indices = torch.topk(
            krum_scores, k=selected_count, largest=False
        ).indices
        aggregate = logits_stack[selected_indices].mean(dim=0)
        active_ids = self._get_active_client_ids(n_uploads)
        selected_ids = [active_ids[index] for index in selected_indices.tolist()]
        mode = "Krum" if selected_count == 1 else "Multi-Krum"
        sys.stdout.write(
            f"[{mode}] f={f} ({f_source}), nearest={neighbor_count}, "
            f"selected client ids={selected_ids}.\n"
        )
        return aggregate, krum_scores


class AFA(LogitAggregationDefence):
    """Adaptive Federated Averaging applied to complete client logit uploads.

    This follows AFA's iterative cosine-similarity filtering and per-client
    Beta reliability posterior. Its permanent block list affects aggregation
    only; clients still train and upload so that the existing runner keeps a
    fixed client schedule.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client_sizes = {
            int(client.client_id): max(1, len(client.trainData))
            for client in self.clients
        }
        self.good_counts = {client_id: 0 for client_id in self.client_sizes}
        self.bad_counts = {client_id: 0 for client_id in self.client_sizes}
        self.blocked_client_ids = set()

    def _quality_probability(self, client_id):
        alpha = float(self._config_value("afa_prior_alpha", 3.0)) + self.good_counts[client_id]
        beta = float(self._config_value("afa_prior_beta", 3.0)) + self.bad_counts[client_id]
        return alpha / (alpha + beta)

    @staticmethod
    def _cosine_similarity_to(aggregate, logits_stack):
        reference = aggregate.reshape(1, -1).float()
        updates = logits_stack.reshape(logits_stack.size(0), -1).float()
        return torch.nn.functional.cosine_similarity(updates, reference, dim=1, eps=1e-12)

    @staticmethod
    def _beta_cdf_at_half(alpha, beta):
        """Exact Beta(alpha, beta) CDF at 0.5 for AFA's count-based posterior."""
        alpha_int, beta_int = int(round(alpha)), int(round(beta))
        if not math.isclose(alpha, alpha_int) or not math.isclose(beta, beta_int):
            raise ValueError("AFA blocking requires integer-valued afa_prior_alpha and afa_prior_beta.")
        n = alpha_int + beta_int - 1
        log_terms = torch.tensor(
            [
                math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) - n * math.log(2.0)
                for j in range(alpha_int, n + 1)
            ],
            dtype=torch.float64,
        )
        return float(torch.exp(torch.logsumexp(log_terms, dim=0)))

    @torch.no_grad()
    def get_aggregated_logits(self, client_logits_list: list):
        logits_stack = torch.stack(client_logits_list)
        n_uploads = logits_stack.size(0)
        active_ids = self._get_active_client_ids(n_uploads)
        active_indices = [
            index for index, client_id in enumerate(active_ids)
            if client_id not in self.blocked_client_ids
        ]
        if not active_indices:
            raise RuntimeError("AFA has blocked every active client upload.")

        xi = float(self._config_value("afa_xi", 2.0))
        delta_xi = float(self._config_value("afa_delta_xi", 0.5))
        if xi < 0.0 or delta_xi < 0.0:
            raise ValueError("afa_xi and afa_delta_xi must be non-negative.")

        good_indices = list(active_indices)
        rejected_indices = []
        while True:
            good_ids = [active_ids[index] for index in good_indices]
            weights = torch.tensor(
                [self._quality_probability(client_id) * self.client_sizes[client_id] for client_id in good_ids],
                device=logits_stack.device,
                dtype=logits_stack.dtype,
            )
            aggregate = (logits_stack[good_indices] * weights.view(-1, 1, 1)).sum(dim=0) / weights.sum()
            similarities = self._cosine_similarity_to(aggregate, logits_stack[good_indices])
            mean_similarity = similarities.mean()
            median_similarity = similarities.median()
            std_similarity = similarities.std(unbiased=False)

            if mean_similarity < median_similarity:
                bad_mask = similarities < median_similarity - xi * std_similarity
            else:
                bad_mask = similarities > median_similarity + xi * std_similarity
            new_rejections = [
                good_indices[offset] for offset in torch.nonzero(bad_mask, as_tuple=False).flatten().tolist()
            ]
            if not new_rejections:
                break
            if len(new_rejections) == len(good_indices):
                raise RuntimeError("AFA marked every remaining client upload as bad.")
            rejected_indices.extend(new_rejections)
            rejected_set = set(new_rejections)
            good_indices = [index for index in good_indices if index not in rejected_set]
            xi += delta_xi

        rejected_set = set(rejected_indices)
        for index, client_id in enumerate(active_ids):
            if index in rejected_set:
                self.bad_counts[client_id] += 1
            elif index in active_indices:
                self.good_counts[client_id] += 1

        block_threshold = float(self._config_value("afa_block_threshold", 0.95))
        if not 0.0 < block_threshold < 1.0:
            raise ValueError("afa_block_threshold must be in (0, 1).")
        newly_blocked = []
        for client_id in [active_ids[index] for index in rejected_indices]:
            alpha = float(self._config_value("afa_prior_alpha", 3.0)) + self.good_counts[client_id]
            beta = float(self._config_value("afa_prior_beta", 3.0)) + self.bad_counts[client_id]
            if self._beta_cdf_at_half(alpha, beta) > block_threshold:
                self.blocked_client_ids.add(client_id)
                newly_blocked.append(client_id)

        final_ids = [active_ids[index] for index in good_indices]
        final_weights = torch.tensor(
            [self._quality_probability(client_id) * self.client_sizes[client_id] for client_id in final_ids],
            device=logits_stack.device,
            dtype=logits_stack.dtype,
        )
        aggregate = (
            logits_stack[good_indices] * final_weights.view(-1, 1, 1)
        ).sum(dim=0) / final_weights.sum()
        outlier_scores = self._mean_logit_distances(logits_stack, aggregate)
        rejected_ids = [active_ids[index] for index in rejected_indices]
        sys.stdout.write(
            f"[AFA] rejected this round={rejected_ids}; newly blocked={newly_blocked}; "
            f"blocked total={sorted(self.blocked_client_ids)}; retained={len(good_indices)}/{n_uploads}.\n"
        )
        return aggregate, outlier_scores


class PredictionMedian(NoDefence):
    """PredictionMedian defence"""

    def get_aggregated_predictions(self, client_predictions_list: list):
        tensor_stack = torch.stack(client_predictions_list)
        median_probabilities = torch.quantile(tensor_stack, 0.5, dim=0, interpolation='nearest')

        # Ensure that the median probabilities sum to 1
        median_probabilities = median_probabilities / median_probabilities.sum(dim=1, keepdim=True)
        outlier_scores = torch.linalg.norm(tensor_stack - median_probabilities, dim=2)
        mean_outlier_scores = torch.mean(outlier_scores, dim=1)
        return median_probabilities, mean_outlier_scores

    def get_aggregated_logits(self, client_logits_list: list):
        tensor_stack = torch.stack(client_logits_list)
        median_logits = torch.quantile(tensor_stack, 0.5, dim=0, interpolation='nearest')
        outlier_scores = torch.linalg.norm(tensor_stack - median_logits, dim=2)
        mean_outlier_scores = torch.mean(outlier_scores, dim=1)
        return median_logits, mean_outlier_scores

class PredictionGeoMedian(NoDefence):
    """PredictionGeoMedian defence"""

    def get_aggregated_predictions(self, client_predictions_list: list):
        pred_stack = torch.stack(client_predictions_list)  # shape: [n_clients, pub_ds_size, n_classes]
        geomedian, outlier_scores = Utils.geomedian(pred_stack)

        # ensure all probs are non-negative
        geomedian = torch.clamp(geomedian, min=0.)
        # normalize to ensure we are in the simplex
        geomedian = torch.nn.functional.normalize(geomedian, p=1.,dim=1)
        assert torch.all(geomedian >= 0), 'GM is unstable, neg probabilities'
        assert torch.allclose(torch.sum(geomedian,dim=1),torch.tensor([1.],device=self.clients[0].device)), 'GM unstable, sum>=1'

        mean_outlier_scores = torch.mean(outlier_scores, dim=0)

        return geomedian, mean_outlier_scores

    def get_aggregated_logits(self, client_logits_list: list):
        logits_stack = torch.stack(client_logits_list)
        geomedian, outlier_scores = Utils.geomedian(logits_stack)
        mean_outlier_scores = torch.mean(outlier_scores, dim=0)
        return geomedian, mean_outlier_scores

class PredictionFilter(NoDefence):
    """PredictionFilter defence"""


    @torch.no_grad()
    def get_aggregated_predictions(self, client_predictions_list: list):
        if self.config["filter_threshold"] is not None:
            assert self.config["filter_quantile"] is None, "Cannot choose both filter_threshold and filter_quantile"
        else:
            assert self.config["filter_quantile"] is not None, "choose either filter_threshold, or filter_quantile"

        pred_stack = torch.stack(client_predictions_list) # shape: [n_clients, pub_ds_size, n_classes]

        # compute outlier scores
        outlier_scores = Utils.filter_outlier_scores(pred_stack)

        # compute filtered mean
        if self.config["filter_threshold"] is not None:
            threshold = self.config["filter_threshold"]
        else:
            threshold = torch.quantile(torch.abs(outlier_scores), self.config["filter_quantile"], dim=1, interpolation='higher')
        filtered_mean = Utils.filtered_mean(pred_stack, outlier_scores, threshold)
        mean_outlier_scores = torch.mean(torch.abs(outlier_scores).squeeze(-1), dim=0)
        return filtered_mean, mean_outlier_scores

    def get_aggregated_logits(self, client_logits_list: list):
        return self.get_aggregated_predictions(client_logits_list)


class Cronus(NoDefence):
    """
    Cronus defence:
    Filter out (alpha N / 2) of predictions, then recompute outlier scores and filter out (alpha N / 2) again
    """

    @torch.no_grad()
    def get_aggregated_predictions(self, client_predictions_list: list):
        pred_stack = torch.stack(client_predictions_list) # shape: [n_clients, pub_ds_size, n_classes]
        max_n_byz = Utils.get_highest_byz(self.config["n_clients"])
        quantile = 1 - (1. * max_n_byz / (2. * self.config["n_clients"]))


        # 1. filtering of predictions
        outlier_scores = Utils.filter_outlier_scores(pred_stack)
        k = int(quantile * pred_stack.size()[0])
        if k <1: k = 1
        _, idx = torch.topk(torch.abs(outlier_scores), k, dim=1,largest=False)  # return k smallest
        reduced_pred = torch.zeros(k, pred_stack.size()[1], pred_stack.size()[2], device=pred_stack.device, dtype=pred_stack.dtype)
        for i in range(pred_stack.size()[1]):
            nonzero = idx[i,:,0]
            reduced_pred[:,i,:] = pred_stack[nonzero,i,:]

        # 2. filtering
        outlier_scores = Utils.filter_outlier_scores(reduced_pred)
        threshold = torch.quantile(torch.abs(outlier_scores), quantile, dim=1, interpolation='higher')
        filtered_mean = Utils.filtered_mean(reduced_pred, outlier_scores, threshold)
        outlier_scores = torch.linalg.norm(pred_stack.permute(1,2,0) - filtered_mean.unsqueeze(-1).to(pred_stack.device),dim=1)
        mean_outlier_scores = torch.mean(outlier_scores, dim=0)
        return filtered_mean, mean_outlier_scores

    def get_aggregated_logits(self, client_logits_list: list):
        return self.get_aggregated_predictions(client_logits_list)

def choose_aggregation_expweights(aggregation_method):
    class ExpWeights(aggregation_method):
        """
        Maintain weights for each client based on outlier score in each communication round
        """
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            assert self.config['exp_stepsize'] is not None, 'speicify stepsize for exp weights'
            self.w = torch.ones(self.config['n_clients']).to(self.clients[0].device)

        def update_weights(self, loss):
            self.w = self.w * torch.exp(-1. * self.config['exp_stepsize'] * loss)
            assert torch.all(self.w >= 0), f'weights need to be non-negative, here: {self.w}'

        def weighted_sum(self, pred_stack):
            p = torch.nn.functional.normalize(self.w, p=1.,dim=0)
            assert torch.all(p >= 0.), f'probabilities have to be >= 0, here: {p}'
            assert torch.all(p <= 1.), f'probabilities have to be <=1, here: {p}'
            assert torch.isclose(torch.abs(torch.sum(p)), torch.tensor([1.]).to(p.device)), f'probabilities have to sum to one, here:{torch.sum(p)}'
            weighted_pred = torch.matmul(pred_stack.permute(1,2,0),p)
            return weighted_pred, p

        @torch.no_grad()
        def get_aggregated_predictions(self, client_predictions_list: list):
            # Compute mean
            pred_stack = torch.stack(client_predictions_list) # shape: [n_clients, pub_ds_size, n_classes]
            mean_pred, mean_outlier_scores = super().get_aggregated_predictions(client_predictions_list)
            self.update_weights(mean_outlier_scores)
            weighted_pred, p = self.weighted_sum(pred_stack)
            return weighted_pred, p

        @torch.no_grad()
        def get_aggregated_logits(self, client_logits_list: list):
            logits_stack = torch.stack(client_logits_list)
            mean_logits, mean_outlier_scores = super().get_aggregated_logits(client_logits_list)
            self.update_weights(mean_outlier_scores)
            weighted_logits, p = self.weighted_sum(logits_stack)
            return weighted_logits, p
    return ExpWeights


