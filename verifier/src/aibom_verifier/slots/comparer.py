class ExactBytesComparer:
    def equal(self, left: bytes, right: bytes) -> bool:
        return left == right
