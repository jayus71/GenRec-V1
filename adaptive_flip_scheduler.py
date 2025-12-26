"""
Adaptive Flip Probability Scheduler for GenRec-V1
Based on Sport dataset characteristics:
- 63.60% users have 3-5 interactions (extremely sparse)
- 34.37% users have 5-20 interactions (moderately sparse)
- 2.03% users have 20+ interactions (relatively dense)
"""

import torch
import numpy as np


class AdaptiveFlipScheduler:
    """
    Three-dimensional adaptive flip probability scheduling:
    1. User activity-based: Different flip probabilities for user groups
    2. Epoch-based: Dynamic scheduling across training epochs
    3. Item popularity-based: Higher flip for long-tail items
    """

    def __init__(self,
                 user_interaction_counts: np.ndarray,
                 item_popularity: np.ndarray,
                 total_epochs: int = 50,
                 base_flip_prob: float = 0.15,
                 use_activity_adaptive: bool = True,
                 use_epoch_adaptive: bool = True,
                 use_popularity_adaptive: bool = True):
        """
        Args:
            user_interaction_counts: (num_users,) array of interaction counts
            item_popularity: (num_items,) array of item interaction counts
            total_epochs: Total training epochs
            base_flip_prob: Base flip probability (default from paper)
            use_activity_adaptive: Enable user activity-based scheduling
            use_epoch_adaptive: Enable epoch-based scheduling
            use_popularity_adaptive: Enable item popularity-based scheduling
        """
        self.user_counts = user_interaction_counts
        self.item_popularity = item_popularity
        self.total_epochs = total_epochs
        self.base_flip_prob = base_flip_prob

        self.use_activity_adaptive = use_activity_adaptive
        self.use_epoch_adaptive = use_epoch_adaptive
        self.use_popularity_adaptive = use_popularity_adaptive

        # User activity thresholds based on Sport dataset statistics
        self.activity_thresholds = {
            'extreme_low': 5,   # 63.60% users (0-5 interactions)
            'low': 10,          # 24.52% users (5-10 interactions)
            'medium': 20,       # 9.85% users (10-20 interactions)
            'high': 50,         # 1.93% users (20-50 interactions)
        }

        # Flip probability multipliers for each user group
        self.activity_multipliers = {
            'extreme_low': 1.5,   # Boost for extremely sparse users
            'low': 1.2,           # Moderate boost
            'medium': 1.0,        # Base probability
            'high': 0.7,          # Reduce for dense users
            'extreme_high': 0.5,  # Minimal flip for super active users
        }

        # Classify users into activity groups
        self._classify_users()

        # Item popularity statistics
        self.item_pop_median = np.median(item_popularity)
        self.item_pop_75percentile = np.percentile(item_popularity, 75)

        print(f"\n=== AdaptiveFlipScheduler Initialized ===")
        print(f"Total users: {len(user_interaction_counts)}")
        print(f"Activity group distribution:")
        for group, count in self.user_group_counts.items():
            pct = count / len(user_interaction_counts) * 100
            print(f"  {group}: {count} users ({pct:.2f}%)")
        print(f"Base flip probability: {base_flip_prob}")
        print(f"Item popularity median: {self.item_pop_median:.2f}")
        print(f"Item popularity 75%: {self.item_pop_75percentile:.2f}")

    def _classify_users(self):
        """Classify users into activity groups based on interaction counts"""
        self.user_groups = np.zeros(len(self.user_counts), dtype=int)

        # 0: extreme_low, 1: low, 2: medium, 3: high, 4: extreme_high
        self.user_groups[self.user_counts <= self.activity_thresholds['extreme_low']] = 0
        self.user_groups[(self.user_counts > self.activity_thresholds['extreme_low']) &
                        (self.user_counts <= self.activity_thresholds['low'])] = 1
        self.user_groups[(self.user_counts > self.activity_thresholds['low']) &
                        (self.user_counts <= self.activity_thresholds['medium'])] = 2
        self.user_groups[(self.user_counts > self.activity_thresholds['medium']) &
                        (self.user_counts <= self.activity_thresholds['high'])] = 3
        self.user_groups[self.user_counts > self.activity_thresholds['high']] = 4

        # Count users in each group
        group_names = ['extreme_low', 'low', 'medium', 'high', 'extreme_high']
        self.user_group_counts = {
            name: np.sum(self.user_groups == i)
            for i, name in enumerate(group_names)
        }

    def get_epoch_multiplier(self, current_epoch: int) -> float:
        """
        Compute epoch-based flip probability multiplier

        Strategy:
        - Early epochs (0-20%): High flip (1.3x) for exploration
        - Mid epochs (20-60%): Gradual decay from 1.3x to 0.8x
        - Late epochs (60-100%): Low flip (0.8x-0.5x) for exploitation
        """
        if not self.use_epoch_adaptive:
            return 1.0

        progress = current_epoch / self.total_epochs

        if progress < 0.2:  # Early stage: exploration
            return 1.3
        elif progress < 0.6:  # Mid stage: gradual transition
            # Linear decay from 1.3 to 0.8
            return 1.3 - (progress - 0.2) / 0.4 * 0.5
        else:  # Late stage: exploitation
            # Linear decay from 0.8 to 0.5
            return 0.8 - (progress - 0.6) / 0.4 * 0.3

    def get_user_flip_probabilities(self,
                                     user_ids: torch.Tensor,
                                     current_epoch: int) -> torch.Tensor:
        """
        Compute adaptive flip probabilities for a batch of users

        Args:
            user_ids: (batch_size,) tensor of user IDs
            current_epoch: Current training epoch

        Returns:
            flip_probs: (batch_size,) tensor of flip probabilities
        """
        batch_size = user_ids.size(0)
        flip_probs = torch.ones(batch_size, device=user_ids.device) * self.base_flip_prob

        if not self.use_activity_adaptive:
            # Only apply epoch multiplier
            epoch_mult = self.get_epoch_multiplier(current_epoch)
            flip_probs = flip_probs * epoch_mult
            return flip_probs

        # Get user groups for this batch
        user_ids_np = user_ids.cpu().numpy()
        batch_groups = self.user_groups[user_ids_np]

        # Apply activity-based multipliers
        group_names = ['extreme_low', 'low', 'medium', 'high', 'extreme_high']
        for i, group_name in enumerate(group_names):
            mask = batch_groups == i
            if np.any(mask):
                multiplier = self.activity_multipliers[group_name]
                flip_probs[mask] *= multiplier

        # Apply epoch-based multiplier
        if self.use_epoch_adaptive:
            epoch_mult = self.get_epoch_multiplier(current_epoch)
            flip_probs = flip_probs * epoch_mult

        # Clip to valid probability range
        flip_probs = torch.clamp(flip_probs, min=0.01, max=0.95)

        return flip_probs

    def get_item_flip_probabilities(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute item popularity-based flip probability multipliers

        Strategy:
        - Long-tail items (below median): 1.2x multiplier
        - Mid-popularity items (median-75%): 1.0x multiplier
        - Popular items (above 75%): 0.8x multiplier
        """
        if not self.use_popularity_adaptive:
            return torch.ones(item_ids.size(0), device=item_ids.device)

        item_ids_np = item_ids.cpu().numpy()
        item_pops = self.item_popularity[item_ids_np]

        multipliers = np.ones_like(item_pops, dtype=np.float32)

        # Long-tail items: boost flip probability
        multipliers[item_pops < self.item_pop_median] = 1.2

        # Popular items: reduce flip probability
        multipliers[item_pops > self.item_pop_75percentile] = 0.8

        return torch.tensor(multipliers, device=item_ids.device)

    def get_adaptive_flip_matrix(self,
                                  user_ids: torch.Tensor,
                                  item_ids: torch.Tensor,
                                  current_epoch: int) -> torch.Tensor:
        """
        Compute fully adaptive flip probability matrix for user-item pairs

        Args:
            user_ids: (batch_size,) tensor of user IDs
            item_ids: (batch_size,) tensor of item IDs
            current_epoch: Current training epoch

        Returns:
            flip_matrix: (batch_size,) tensor of adaptive flip probabilities
        """
        # Get user-based flip probabilities
        user_flip_probs = self.get_user_flip_probabilities(user_ids, current_epoch)

        # Get item-based multipliers
        item_multipliers = self.get_item_flip_probabilities(item_ids)

        # Combine user and item factors
        flip_probs = user_flip_probs * item_multipliers

        # Clip to valid range
        flip_probs = torch.clamp(flip_probs, min=0.01, max=0.95)

        return flip_probs

    def print_epoch_stats(self, current_epoch: int):
        """Print statistics for current epoch"""
        epoch_mult = self.get_epoch_multiplier(current_epoch)

        print(f"\n=== Epoch {current_epoch} Flip Probability Schedule ===")
        print(f"Epoch multiplier: {epoch_mult:.3f}")
        print(f"Activity-based flip probabilities:")

        group_names = ['extreme_low', 'low', 'medium', 'high', 'extreme_high']
        for i, group_name in enumerate(group_names):
            base_mult = self.activity_multipliers[group_name]
            final_prob = self.base_flip_prob * base_mult * epoch_mult
            final_prob = np.clip(final_prob, 0.01, 0.95)
            count = self.user_group_counts[group_name]
            pct = count / len(self.user_counts) * 100
            print(f"  {group_name:15s}: {final_prob:.4f} ({count:5d} users, {pct:5.2f}%)")


def build_scheduler_from_dataset(args, handler) -> AdaptiveFlipScheduler:
    """
    Build AdaptiveFlipScheduler from DataHandler

    Args:
        args: Argument namespace from Params.py
        handler: DataHandler instance

    Returns:
        scheduler: Initialized AdaptiveFlipScheduler
    """
    # Compute user interaction counts from training matrix
    trnMat = handler.trnMat
    user_counts = np.array(trnMat.sum(axis=1)).flatten()

    # Compute item popularity from training matrix
    item_popularity = np.array(trnMat.sum(axis=0)).flatten()

    # Build scheduler
    scheduler = AdaptiveFlipScheduler(
        user_interaction_counts=user_counts,
        item_popularity=item_popularity,
        total_epochs=args.epoch,
        base_flip_prob=args.flip_prob if hasattr(args, 'flip_prob') else 0.15,
        use_activity_adaptive=args.use_activity_adaptive if hasattr(args, 'use_activity_adaptive') else True,
        use_epoch_adaptive=args.use_epoch_adaptive if hasattr(args, 'use_epoch_adaptive') else True,
        use_popularity_adaptive=args.use_popularity_adaptive if hasattr(args, 'use_popularity_adaptive') else True,
    )

    return scheduler
