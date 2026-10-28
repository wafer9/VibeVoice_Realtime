# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The inference entrypoint also partitions inputs under torchrun."""
from twinlakes.bin.infer import main

if __name__ == '__main__':
    main()
