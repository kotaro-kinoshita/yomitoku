class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        # 経路圧縮
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return False

        # union by size（小さい木を大きい木へ）
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        return True

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def group_size(self, x):
        return self.size[self.find(x)]

    def groups(self):
        # {root: [members]}
        res = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            res.setdefault(r, []).append(i)
        return list(res.values())

    def group_id(self):
        # [group_id for each element]
        res = [-1] * len(self.parent)
        gid = 0
        for i in range(len(self.parent)):
            r = self.find(i)
            if res[r] == -1:
                res[r] = gid
                gid += 1
            res[i] = res[r]
        return res
