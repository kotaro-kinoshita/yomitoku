class UnionFind:
    """Unionfind

    ノードのグループ化, 同一グループに属しているかの判定
    計算量: O(α(n)) log n より早い

    """

    def __init__(self, n):
        """

        Args:
            n (int): ノードの総数
        """
        self.parents = [i for i in range(n)]
        self.rank = [0] * n

    def find(self, x):
        """親が何か探す

        Args:
            x (int): 親を探したいノード番号

        Returns:
            int: 親のノード番号
        """
        if self.parents[x] == x:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        """グループの併合

        Args:
            x, y (int): 併合したいノードの番号
        """
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if self.rank[x] < self.rank[y]:
            self.parents[x] = y
        else:
            self.parents[y] = x
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1

    def same(self, x, y):
        """同じグループに属しているか判定

        Args:
            x, y (int): 同じグループに属しているか判定したいノードの番号

        Returns:
            bool: 同じグループならTrue, そうでないならFalse
        """
        return self.find(x) == self.find(y)

    def new_node(self):
        """新しいノードを追加する"""

        self.parents.append(len(self.parents))
        self.rank.append(0)
        return len(self.parents) - 1
