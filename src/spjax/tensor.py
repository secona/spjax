class SparseTensor:
    @classmethod
    def from_file(cls, filename):
        try:
            import scipy
        except:
            raise ImportError("Failed to import SciPy for reading sparse tensor")

        m_coo = scipy.io.mmread(filename)
        print("nnz: ", m_coo.nnz)

