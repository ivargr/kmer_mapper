import numpy as np
from .numpyutil import cumcount
import logging


class ModuloHashLookup:
    def __init__(self, values, mod, n_entries, lookup_end):
        self._values = values # np.append(sorted_values, np.array([4**31], dtype=np.uint64))
        self._mod = mod
        self._n_entries = n_entries
        self._lookup_end = lookup_end

    def _get_hash(self, queries):
        return queries % self._mod

    def get_hit_idxs(self, queries):
        hashes = self._get_hash(queries)
        return np.flatnonzero(self._n_entries[hashes])

    def get_hits(self, queries):
        queries = queries[self.get_hit_idxs(queries)]
        indexes = self.get_index(queries)
        return indexes[self._values[indexes]==queries]

    def get_index(self, queries):
        hashes = self._get_hash(queries)
        n_entries = self._n_entries[hashes]
        n_iters = np.floor(np.log2(n_entries)).astype(int)+1
        n_iter_sort_args = np.argsort(n_iters)
        n_entries = n_entries[n_iter_sort_args]
        hashes = hashes[n_iter_sort_args]
        n_iters = n_iters[n_iter_sort_args]
        R = self._lookup_end[hashes]
        L = R - n_entries
        found_indices = np.zeros_like(queries, dtype=int)
        found_indices[n_iter_sort_args] = self._binary_search(queries[n_iter_sort_args], L, R, n_iters)
        return found_indices

    def _binary_search(self, queries, L, R, n_iters):
        """
        https://en.wikipedia.org/wiki/Binary_search_algorithm#Alternative_procedure
        Look for queries in range [L, R>
        Runs only reuqired iterations for each group
        """
        boundries = np.vstack((L, R))
        indexes = np.arange(boundries.shape[-1])
        stops = np.cumsum(np.bincount(n_iters))
        for stop in stops[:-1]:
            m = (boundries[:, stop:].sum(axis=0))//2
            is_larger = (self._values[m] > queries[stop:]).view(np.uint8)
            boundries[is_larger, indexes[stop:]] = m
        return boundries[0]

    @classmethod
    def from_values_and_mod(cls, keys, modulo):
        hashes = (keys % modulo).astype(int)
        counts = np.bincount(hashes, minlength=modulo)
        args = np.lexsort([keys, hashes])
        hash_sorted = keys[args]
        n_entries=counts
        lookup_end = np.cumsum(counts)
        return cls(hash_sorted, modulo, n_entries, lookup_end)

    @classmethod
    def from_file(cls, filename):
        D = np.load(filename)
        return cls(D["values"], D["mod"], D["n_entries"], D["lookup_end"])

    def to_file(self, filename):
        np.savez(filename, 
                 values=self._values,
                 lookup_end=self._lookup_end,
                 mod=self._mod,
                 n_entries=self._n_entries)


class ComplicatedModuloHashLookup:
    bitshift=np.uint64(64-27)
    n_entries_mask = np.uint64(2**4-1)
    def __init__(self, value_arrays, mod, n_entries, local_idxs):
        self._value_arrays = value_arrays
        self.n_kmers = sum(v.size for v in value_arrays)
        print([a.shape[0] for a in self._value_arrays])
        self._entry_offsets = np.cumsum([v.size for v in self._value_arrays])
        # print([(i, a.size) for i, a in enumerate(self._value_arrays)])
        # self._n_entries = n_entries
        # self._local_idxs = local_idxs
        # self._table = np.hstack((n_entries[:, None], local_idxs[:, None]))
        for v in value_arrays[1:]:
            print(v.dtype)
            v >>= np.uint64(27)
        print(local_idxs.dtype)
        self._codes = (local_idxs.astype(np.uint64)<<self.bitshift)+np.uint64(n_entries)# np.hstack((n_entries[:, None], local_idxs[:, None]))
        print(self._codes.dtype)
        self._mod = mod

    def _local_get_hits(self, values, queries):
        return np.nonzero(values==queries[:, None])

    def _get_subsets(self, hashes):
        codes = self._codes[hashes]
        return codes & np.uint64(2**self.bitshift-1), codes>>self.bitshift
    # return self._table[hashes]
        # return (self._n_entries[hashes], self._local_idxs[hashes])
    def _get_hashes(self, queries):
        return queries % self._mod

    def _super_local_entries(self, n_entries, local_idxs, queries, i):
        idxs = np.flatnonzero(n_entries==i)
        return local_idxs[idxs], queries[idxs]

    def _get_n_entries(self, codes):
        return codes & np.uint64(2**self.bitshift-1)

    def _split_codes(self, codes):
        return codes & np.uint64(2**self.bitshift-1), codes>>self.bitshift

    def _super_local_codes(self, codes, n_entries, i):
        local_codes = codes[n_entries==i]
        superlocal = local_codes >> self.bitshift
        local_codes &= np.uint64(2**27-1)
        return superlocal, local_codes


    def __get_codes(self, queries):
        return self._codes[self._get_hashes(queries)]
    
    def prelim(self, queries):
        #hashes = self._get_hashes(queries)        
        codes = self.__get_codes(queries)
        n_entries = codes.astype(np.uint8)#  & self.n_entries_mask
        codes &= ~self.n_entries_mask
        codes |= queries >> 27
        return n_entries, codes

    def summarize(self, v_idx, offsets):
        sizes = [v.size for v in v_idx]
        return np.concatenate([v.view(int)*i+e for i, (v, e) in enumerate(zip(v_idx, self._entry_offsets), 1)])+np.concatenate(offsets)

    def get_hits(self, queries):
        #hashes = self._get_hashes(queries)        
        # codes = 
        # m = np.flatnonzero(self._get_n_entries(codes))
        # codes = codes[m]
        # queries = queries[m]
        #codes = self._codes[hashes]
        #n_entries = codes & self.n_entries_mask
        # print(codes.dtype, n_entries.dtype, self.n_entries_mask.dtype)
        #codes &= ~self.n_entries_mask
        #codes |= queries >> 27
        n_entries, codes = self.prelim(queries)
        # n_entries, local_idxs = self._split_codes(self._codes[hashes])# codes)
        # self._get_subsets(hashes)
        matches = []
        N = int(np.max(n_entries)+1)
        # print(n_entries.dtype)
        v_idx=[]
        offsets=[]
        for i in range(1, N):
            #idxs = np.flatnonzero(n_entries==i)
            #superlocal = local_idxs[idxs]
            #superlocal, q = self._super_local_entries(n_entries, local_idxs, queries, i)
            # local_codes = codes[np.flatnonzero(n_entries==i)]
            # superlocal = local_codes >> self.bitshift
            # local_codes &= np.uint64(2**27-1)
            superlocal, local_codes = self._super_local_codes(codes, n_entries, i)
            k, offset = self._local_get_hits(self._value_arrays[i][superlocal], local_codes)# queries[idxs])
            # print(k.dtype, offset.dtype, superlocal.dtype)
            v_idx.append(superlocal[k])
            offsets.append(offset)
            #matches.append((superlocal[k]*i+offset).astype(int))
        return self.summarize(v_idx, offsets)
    #return np.concatenate(matches)

    @classmethod
    def from_file(cls, filename):
        D = np.load(filename)
        n_entries = D["n_entries"]
        N = np.max(n_entries)+1
        local_idxs = D["local_idxs"]
        value_array = [D[f"V{i}"] for i in range(N)]
        return cls(value_array, D["mod"], n_entries, local_idxs)


    @classmethod
    def from_old_file(cls, filename):
        D = np.load(filename)
        n_entries = D["n_entries"]
        values = D["values"]
        N = np.max(n_entries)+1
        # value_arrays = [np.array([])] + [values[n_entries==i] for i in range(1, N)]
        local_offsets = cumcount(n_entries)
        value_arrays = [np.array([])]
        lookup_ends = D["lookup_end"]
        for i in range(1, N):
            idxs = np.flatnonzero(n_entries==i)
            value_array = np.empty((idxs.size, i), dtype=np.uint64)
            ends = lookup_ends[idxs]
            for j in range(1, i+1):
                value_array[:, -j] = values[ends-j]
            value_arrays.append(value_array)

        return cls(value_arrays, D["mod"], n_entries, local_offsets)
        value_arrays = [list() for _ in range(N)]
        local_offsets = []
        i = 0
        for n_entry, lookup_end in zip(n_entries, D["lookup_end"]):
            if i % 1000000==0:
                print(i, len(n_entries))
            i += 1
            if n_entry==0:
                local_offsets.append(0)
                continue
            local_offsets.append(len(value_arrays[n_entry]))
            v = values[lookup_end-n_entry:lookup_end]
            assert v.size == n_entry, (v, lookup_end, n_entry)
            value_arrays[n_entry].append(v)

        print("Finished")
        mod = D["mod"]
        D = None
        values = None
        local_offsets = np.array(local_offsets)
        for i in range(N):
            value_arrays[i] = np.array(value_arrays[i])
        return cls(value_arrays, mod, n_entries, local_offsets)
        
    def to_file(self, filename):
        np.savez(filename, 
                 n_entries=self._n_entries,
                 mod=self._mod,
                 local_idxs = self._local_idxs,
                 **{f"V{i}": va for i, va in enumerate(self._value_arrays)})

class SimpleModuloHashLookup(ModuloHashLookup):
    def get_index(self, queries):
        hashes = self._get_hash(queries)
        n_entries = self._n_entries[hashes]
        lookup_end = self._lookup_end[hashes]
        result = np.empty_like(hashes, dtype=np.int32)
        for i in range(1, np.max(n_entries)+1):
            n_entries -= 1
            idxs = np.flatnonzero(n_entries)
            result[idxs] = np.argmax(hashes[idxs] == (self._values[lookup_end[idxs]-np.arange(1, i+1)[:, None]]), axis=0)

        return lookup_end-result-1
        # print(all_values.shape, np.argmax(hashes==all_values, axis=0).shape)
        return lookup_end - np.argmax(hashes==all_values, axis=0)-1
        is_equal = np.any(hashes==all_values, axis=0)
        return lookup_end[is_equal] - np.argmax(hashes[is_equal]==all_values[:, is_equal], axis=0) - 1

    def _get_index(self, queries):
        hashes = self._get_hash(queries)

        n_iter = np.floor(np.log2(np.max(n_entries))).astype(int)+1
        R = self._lookup_end[hashes]
        L = R - n_entries
        return self._binary_search(queries, L, R, n_iter)

    def _binary_search(self, queries, L, R, n_iter):
        """
        https://en.wikipedia.org/wiki/Binary_search_algorithm#Alternative_procedure
        Look for queries in range [L, R>
        Runs only reuqired iterations for each group
        """
        boundries = np.vstack((L, R))
        indexes = np.arange(boundries.shape[-1])
        m = np.add(boundries[0], boundries[1])
        m//=2
        is_larger = (self._values[m] > queries)
        for _ in range(n_iter-1):
            is_larger = (self._values[m]>queries).view(np.uint8)
            boundries[is_larger, indexes] = m
            np.add(boundries[0], boundries[1], out=m)
            m //= 2
        is_larger = (self._values[m] > queries).view(np.uint8)
        boundries[is_larger, indexes] = m
        return boundries[0]



class NodeCount:
    dtype=np.uint64
    k=31
    lookup_class = ModuloHashLookup
    n_bins = 200000003

    def __init__(self, kmers, kmer_indexes, node_ids):
        self._kmers = kmers
        self._node_ids = node_ids
        self._kmer_indexes = kmer_indexes

    def max_node_id(self):
        return np.max(self._node_ids)

    def index_kmers(self):
        self._indexed_lookup = self.lookup_class.from_values_and_mod(
            self._kmers, np.uint64(self.n_bins))
        self._kmer_indexes = self._indexed_lookup.get_index(self._kmer_indexes)

    def get_node_counts(self, kmers):
        counts = self.count_kmers(kmers)
        return self.get_node_counts_from_kmer_counts(counts)
        # return np.bincount(self._node_ids, counts[self._kmer_indexes])

    def get_node_counts_from_kmer_counts(self, kmer_counts):
        return np.bincount(self._node_ids, kmer_counts[self._kmer_indexes])

    def count_kmers(self, kmers):
        indices = self._indexed_lookup.get_hits(kmers)
        # indices = self._indexed_lookup.find_matches(kmers)
        return np.bincount(indices, minlength=self._kmers.size)

    @classmethod
    def from_old_file(cls, filename):
        D = np.load(filename)
        kmers = D["kmers"]
        repr_kmers = kmers[D["representative_kmers"]]
        obj = cls(kmers, repr_kmers, D["lookup"])
        obj.index_kmers()
        return obj

    @classmethod
    def from_file(cls, filename):
        D = np.load(filename)
        obj = cls(D["kmers"], D["kmer_indexes"], D["node_ids"])
        obj._indexed_lookup=cls.lookup_class(
            D["kmers"], D["mod"], D["n_entries"], D["lookup_end"])
        return obj

    @classmethod
    def from_old_index_files(cls, filename):
        logging.info("From old index files")
        data = np.load(filename)
        kmers = data["kmers"]
        unique_kmers = np.unique(kmers)
        nodes = data["nodes"]
        k = cls(unique_kmers, kmers, nodes)
        k.index_kmers()
        return k

    def to_file(self, filename):
        np.savez(filename, 
                 kmers=self._indexed_lookup._values,
                 kmer_indexes=self._kmer_indexes,
                 node_ids=self._node_ids,
                 lookup_end=self._indexed_lookup._lookup_end,
                 mod=self._indexed_lookup._mod,
                 n_entries=self._indexed_lookup._n_entries)
    
