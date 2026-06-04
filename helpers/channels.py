import math
import random

from .bit import Bit, xor


class BinarySymmetricChannel:
    def __init__(self, epsilon: float) -> None:
        if not (0 <= epsilon <= 1):
            raise ValueError("Epsilon should be between 0 and 1.")
        self.epsilon = epsilon

    def transmit(self, x: Bit) -> Bit:
        """Flip the bit with probability epsilon, otherwise pass it through."""
        if random.uniform(0, 1) < self.epsilon:
            return 1 - x
        return x

    def transition_probability(self, x: Bit, y: Bit) -> float:
        """Return the probability of receiving y given that x was sent."""
        if x == y:
            return 1 - self.epsilon
        return self.epsilon

    def bhattacharyya(self) -> float:
        """
        Compute the Bhattacharyya parameter Z.

        Z measures how similar the output distributions are when sending 0 vs 1.
        Z ≈ 1 means the decoder cannot distinguish them (bad channel).
        Z ≈ 0 means the distributions are very different (reliable channel).
        """
        z: float = 0.0
        for y in (0, 1):
            prob_when_zero: float = self.transition_probability(0, y)
            prob_when_one: float = self.transition_probability(1, y)
            z += math.sqrt(prob_when_zero * prob_when_one)
        return z

    def binary_entropy(self) -> float:
        """Return the binary entropy of the channel's error probability."""
        if self.epsilon == 0 or self.epsilon == 1:
            return 0.0
        return -(
            self.epsilon * math.log2(self.epsilon)
            + (1 - self.epsilon) * math.log2(1 - self.epsilon)
        )

    def symmetric_capacity(self) -> float:
        """Return the symmetric channel capacity (mutual information between input and output)."""
        return 1 - self.binary_entropy()


class CombinedChannel:
    def __init__(self, original_channel: BinarySymmetricChannel) -> None:
        self.channel = original_channel

    def transition_probability(self, y1: Bit, y2: Bit, u1: Bit, u2: Bit) -> float:
        """
        Compute the combined transition probability.

        W2(y1, y2 | u1, u2) = W(y1 | u1 ⊕ u2) * W(y2 | u2)
        """
        x1: Bit = xor(u1, u2)
        x2: Bit = u2
        return self.channel.transition_probability(
            y1, x1
        ) * self.channel.transition_probability(y2, x2)

    def transmit(self, x: Bit) -> Bit:
        return self.channel.transmit(x)

    def bhattacharyya(self) -> float:
        return self.channel.bhattacharyya()

    def minus(self) -> float:
        """
        Compute the Bhattacharyya parameter for the synthesized W⁻ channel.

        W⁻ marginalizes over u2, increasing uncertainty:
        Z(W⁻) = 2·Z - Z²
        """
        return (2 * self.bhattacharyya()) - (self.bhattacharyya() ** 2)

    def plus(self) -> float:
        """
        Compute the Bhattacharyya parameter for the synthesized W⁺ channel.

        W⁺ conditions on u1, decreasing uncertainty:
        Z(W⁺) = Z²
        """
        return self.bhattacharyya() ** 2
