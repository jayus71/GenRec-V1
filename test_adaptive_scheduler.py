"""
Test script for Adaptive Flip Probability Scheduler
Usage: python test_adaptive_scheduler.py
"""

import torch
import numpy as np
from adaptive_flip_scheduler import AdaptiveFlipScheduler

def test_scheduler():
    """Test the adaptive flip scheduler with Sport dataset characteristics"""

    print("=" * 70)
    print("Testing Adaptive Flip Scheduler")
    print("=" * 70)

    # Simulate Sport dataset user interaction distribution
    # 63.60% extreme_low (3-5), 24.52% low (5-10), 9.85% medium (10-20),
    # 1.93% high (20-50), 0.10% extreme_high (50+)
    num_users = 35598

    # Generate user interaction counts following Sport distribution
    user_counts = np.concatenate([
        np.random.randint(3, 6, size=int(num_users * 0.636)),    # extreme_low
        np.random.randint(5, 11, size=int(num_users * 0.245)),   # low
        np.random.randint(10, 21, size=int(num_users * 0.098)),  # medium
        np.random.randint(20, 51, size=int(num_users * 0.019)),  # high
        np.random.randint(50, 101, size=int(num_users * 0.002)), # extreme_high
    ])

    # Pad to exact num_users
    if len(user_counts) < num_users:
        user_counts = np.concatenate([user_counts, np.random.randint(3, 6, size=num_users - len(user_counts))])
    user_counts = user_counts[:num_users]

    # Generate item popularity (simulated)
    num_items = 18357
    item_popularity = np.random.exponential(scale=6.0, size=num_items)  # Long-tail distribution

    # Initialize scheduler
    scheduler = AdaptiveFlipScheduler(
        user_interaction_counts=user_counts,
        item_popularity=item_popularity,
        total_epochs=50,
        base_flip_prob=0.15,
        use_activity_adaptive=True,
        use_epoch_adaptive=True,
        use_popularity_adaptive=True
    )

    print("\n" + "=" * 70)
    print("Test 1: User Activity-Based Flip Probabilities")
    print("=" * 70)

    # Test different user groups
    test_user_ids = torch.tensor([0, 23000, 32000, 35000, 35590])  # Different activity levels
    for epoch in [0, 24, 49]:
        print(f"\nEpoch {epoch}:")
        flip_probs = scheduler.get_user_flip_probabilities(test_user_ids, epoch)
        for i, user_id in enumerate(test_user_ids):
            user_count = user_counts[user_id]
            print(f"  User {user_id:5d} ({user_count:3d} interactions): flip_prob = {flip_probs[i]:.4f}")

    print("\n" + "=" * 70)
    print("Test 2: Epoch-Based Scheduling (Multipliers)")
    print("=" * 70)

    for epoch in [0, 5, 10, 20, 30, 40, 49]:
        mult = scheduler.get_epoch_multiplier(epoch)
        progress = epoch / scheduler.total_epochs * 100
        print(f"Epoch {epoch:2d} ({progress:5.1f}% progress): multiplier = {mult:.4f}")

    print("\n" + "=" * 70)
    print("Test 3: Item Popularity-Based Flip Probabilities")
    print("=" * 70)

    # Test items with different popularity levels
    sorted_items = np.argsort(item_popularity)
    test_item_ids = torch.tensor([
        sorted_items[100],           # Long-tail
        sorted_items[num_items // 2],  # Median
        sorted_items[int(num_items * 0.75)],  # 75th percentile
        sorted_items[-100]           # Popular
    ])

    item_multipliers = scheduler.get_item_flip_probabilities(test_item_ids)
    for i, item_id in enumerate(test_item_ids):
        pop = item_popularity[item_id]
        print(f"  Item {item_id:5d} (popularity={pop:6.2f}): multiplier = {item_multipliers[i]:.4f}")

    print("\n" + "=" * 70)
    print("Test 4: Combined Adaptive Flip Matrix")
    print("=" * 70)

    # Test combined scheduling
    batch_user_ids = torch.tensor([0, 23000, 32000, 35000])  # Different user groups
    batch_item_ids = torch.tensor([
        sorted_items[100],           # Long-tail
        sorted_items[num_items // 2],  # Median
        sorted_items[int(num_items * 0.75)],  # 75th percentile
        sorted_items[-100]           # Popular
    ])

    for epoch in [0, 25, 49]:
        print(f"\nEpoch {epoch}:")
        adaptive_probs = scheduler.get_adaptive_flip_matrix(
            batch_user_ids, batch_item_ids, epoch
        )
        for i in range(len(batch_user_ids)):
            user_count = user_counts[batch_user_ids[i]]
            item_pop = item_popularity[batch_item_ids[i]]
            print(f"  User {batch_user_ids[i]:5d} ({user_count:3d} int.) × "
                  f"Item {batch_item_ids[i]:5d} (pop={item_pop:5.1f}): "
                  f"flip_prob = {adaptive_probs[i]:.4f}")

    print("\n" + "=" * 70)
    print("Test 5: Scheduler Statistics at Different Epochs")
    print("=" * 70)

    for epoch in [0, 10, 25, 40, 49]:
        scheduler.print_epoch_stats(epoch)

    print("\n" + "=" * 70)
    print("All Tests Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    test_scheduler()
