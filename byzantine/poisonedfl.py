"""PoisonedFL-inspired output-space controller, not the paper's parameter attack."""

import math

import torch


class PoisonedFLLogitController:
    """Coordinate malicious uploads along one sample-indexed, centered direction.

    Strength is an absolute per-row RMS offset in raw-logit units. Optional
    adaptation observes only successive broadcast-server outputs, never retained
    client IDs, honest outputs, probe metadata, or ground-truth labels.
    """

    def __init__(self, seed, strength=1.0, direction="fixed", adaptive=False,
                 min_strength=0.01, max_strength=10.0, feedback_interval=5,
                 feedback_target=0.55, adjust_factor=1.25):
        self.seed = int(seed) % (2 ** 63 - 1)
        self.strength = float(strength)
        self.direction_mode = str(direction).lower()
        if not isinstance(adaptive, bool):
            raise ValueError("poisonedfl_adaptive must be a boolean, not a string.")
        self.adaptive = adaptive
        self.min_strength = float(min_strength)
        self.max_strength = float(max_strength)
        self.feedback_interval = int(feedback_interval)
        self.feedback_target = float(feedback_target)
        self.adjust_factor = float(adjust_factor)
        numbers = (self.strength, self.min_strength, self.max_strength,
                   self.feedback_target, self.adjust_factor)
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError("PoisonedFL settings must be finite.")
        if not 0 < self.min_strength <= self.strength <= self.max_strength:
            raise ValueError("PoisonedFL requires 0 < min_strength <= strength <= max_strength.")
        if self.direction_mode not in ("fixed", "fresh"):
            raise ValueError("poisonedfl_direction must be fixed or fresh.")
        if self.feedback_interval < 1 or not 0.5 < self.feedback_target < 1.0:
            raise ValueError("PoisonedFL needs interval >= 1 and 0.5 < feedback_target < 1.")
        if self.adjust_factor <= 1:
            raise ValueError("poisonedfl_adjust_factor must exceed 1.")
        self._shape = None
        self._fixed_direction = None
        self._last_round = None
        self._previous_reference = None
        self._previous_direction = None
        self._aligned = 0
        self._observed = 0
        self._feedback_rounds = 0

    def _direction(self, sample_ids, n_classes, round_number):
        # CPU-local RNG makes attack directions independent of training RNG,
        # dataloader order, malicious-client count, and the chosen device.
        if self.direction_mode == "fixed" and self._fixed_direction is not None:
            canonical = self._fixed_direction
        else:
            salt = 0 if self.direction_mode == "fixed" else int(round_number) * 1000003
            generator = torch.Generator(device="cpu")
            generator.manual_seed((self.seed + salt) % (2 ** 63 - 1))
            canonical = torch.randint(
                0, 2, (len(sample_ids), n_classes), generator=generator,
            ).to(torch.float64).mul_(2).sub_(1)
            # An all-equal row would only shift softmax by a constant.
            degenerate = canonical.eq(canonical[:, :1]).all(dim=1)
            canonical[degenerate, -1] = -canonical[degenerate, 0]
            canonical -= canonical.mean(dim=1, keepdim=True)
            canonical /= canonical.square().mean(dim=1, keepdim=True).sqrt()
            if self.direction_mode == "fixed":
                self._fixed_direction = canonical
        return canonical[sample_ids]

    @staticmethod
    def _validate_logits(logits, shape=None):
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("PoisonedFL expects [public_samples, classes >= 2] logits.")
        if not logits.is_floating_point() or not torch.isfinite(logits).all():
            raise ValueError("PoisonedFL logits must be finite floating-point tensors.")
        if shape is not None and tuple(logits.shape) != shape:
            raise ValueError("PoisonedFL requires aligned public-sample/class shapes.")

    def _feedback(self, reference, previous_direction):
        delta = reference - self._previous_reference
        delta -= delta.mean(dim=1, keepdim=True)
        if not torch.isfinite(delta).all():
            raise ValueError("PoisonedFL feedback difference is not finite.")
        valid = delta.abs() > 1e-7
        count = int(valid.sum())
        aligned = int(((delta * previous_direction > 0) & valid).sum())
        self._aligned += aligned
        self._observed += count
        self._feedback_rounds += 1
        agreement = aligned / count if count else None
        action = "hold"
        if self._feedback_rounds >= self.feedback_interval:
            if self._observed:
                if self._aligned / self._observed >= self.feedback_target:
                    self.strength = min(self.max_strength, self.strength * self.adjust_factor)
                    action = "grow"
                else:
                    self.strength = max(self.min_strength, self.strength / self.adjust_factor)
                    action = "shrink"
            self._aligned = self._observed = self._feedback_rounds = 0
        return agreement, action

    @torch.no_grad()
    def poison(self, malicious_logits, round_number, reference_logits=None, sample_ids=None):
        """Return new malicious tensors; sample IDs must permute the full public set."""
        if not malicious_logits:
            return [], {"malicious_clients": 0}
        first = malicious_logits[0]
        self._validate_logits(first)
        shape = tuple(first.shape)
        for logits in malicious_logits[1:]:
            self._validate_logits(logits, shape)
        if self._shape is not None and shape != self._shape:
            raise ValueError("PoisonedFL public dataset/class shape changed during the run.")
        round_number = int(round_number)
        if self._last_round is not None and round_number <= self._last_round:
            raise ValueError("PoisonedFL attack calls require strictly increasing round numbers.")
        n_samples, n_classes = shape
        if sample_ids is None:
            ids = torch.arange(n_samples)
        else:
            ids = torch.as_tensor(sample_ids, device="cpu")
            if ids.ndim != 1 or ids.is_floating_point() or ids.dtype == torch.bool:
                raise ValueError("PoisonedFL sample IDs must be an integer permutation.")
            ids = ids.to(torch.int64)
        if len(ids) != n_samples or not torch.equal(ids.sort().values, torch.arange(n_samples)):
            raise ValueError("PoisonedFL sample IDs must permute all public indices exactly once.")

        reference = None
        if self.adaptive:
            if reference_logits is None:
                raise ValueError("Adaptive PoisonedFL needs broadcast-server public logits.")
            self._validate_logits(reference_logits, shape)
            reference = reference_logits.detach().to(device="cpu", dtype=torch.float64)
            # Store history in canonical ID order, not current row order.
            reference = reference[ids.argsort()].clone()
        if not n_samples:
            return [x.clone() for x in malicious_logits], {
                "malicious_clients": len(malicious_logits), "strength": self.strength,
                "offset_rms": 0.0, "direction_cosine": None,
                "feedback_alignment": None, "adjustment": "hold", "top1_change": 0.0,
            }

        direction = self._direction(ids, n_classes, round_number)
        canonical_direction = direction[ids.argsort()]
        agreement, action = None, "hold"
        if self.adaptive and self._previous_reference is not None:
            agreement, action = self._feedback(reference, self._previous_direction)

        result = []
        changed = 0
        squared_offset_sum = 0.0
        offsets = {}
        for logits in malicious_logits:
            # Accumulate in float64 for double inputs, float32 for half/single.
            work_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
            key = (logits.device, work_dtype)
            if key not in offsets:
                offsets[key] = direction.to(device=logits.device, dtype=work_dtype) * self.strength
            offset = offsets[key]
            poisoned = (logits.detach().to(work_dtype) + offset).to(logits.dtype)
            if not torch.isfinite(poisoned).all():
                raise ValueError("PoisonedFL offset overflows the upload dtype; reduce strength.")
            result.append(poisoned)
            changed += int((poisoned.argmax(dim=1) != logits.argmax(dim=1)).sum())
            actual_offset = poisoned.to(work_dtype) - logits.detach().to(work_dtype)
            squared_offset_sum += float(actual_offset.double().square().sum())
        cosine = None
        if self._previous_direction is not None:
            cosine = float(torch.nn.functional.cosine_similarity(
                canonical_direction.reshape(1, -1), self._previous_direction.reshape(1, -1),
            ))
        self._shape = shape
        self._last_round = round_number
        self._previous_direction = canonical_direction.clone()
        self._previous_reference = reference
        return result, {
            "malicious_clients": len(result), "strength": self.strength,
            "offset_rms": math.sqrt(squared_offset_sum / (n_samples * n_classes * len(result))),
            "direction_cosine": cosine,
            "feedback_alignment": agreement, "adjustment": action,
            "top1_change": changed / (n_samples * len(result)),
        }
