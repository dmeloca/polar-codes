from helpers.channels import BinarySymmetricChannel, CombinedChannel

channel = CombinedChannel(BinarySymmetricChannel(0.3))
print(f"Z(W^-) = {channel.minus()}")
print(f"Z(W^+) = {channel.plus()}")
print(f"Z(W) = {channel.bhattacharyya()}")
