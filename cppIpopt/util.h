#include <numeric>
#include <Eigen/Core>

/**
 * Calculate the Size of a matrix defined by compact compressed column storage (CCS) format
 *      The Compact format is defined by casadi: https://web.casadi.org/docs/#sec-c-api
 *  
 */
template<typename T>
inline T compCCS_nnz(const T CompCCSFormat[])
{
    return (CompCCSFormat[2] == 1) ? // dense
               CompCCSFormat[0] * CompCCSFormat[1]
                                    : CompCCSFormat[2 + CompCCSFormat[1]]; //sparse
}

/**
 * Get all the triplet representation of a sparse matrix
 *      The Compact format is defined by casadi: https://web.casadi.org/docs/#sec-c-api
 *      The triplet representation is defined by IPOPT: https://coin-or.github.io/Ipopt/IMPL.html#TRIPLET
 */
template<typename T, typename K>
void compCCS_Triplet(const T CompCCSFormat[], K* iRow, K* jCol)
{
    T nnz = compCCS_nnz(CompCCSFormat);
    size_t wp = 0; // write ptr
    if(CompCCSFormat[2]==1){ //dense
        for(K i = 0; i< CompCCSFormat[0]; i++){ // row first arrangement
            for(K j = 0; j< CompCCSFormat[1]; j++){
                iRow[wp] = i;
                jCol[wp] = j;
                ++wp;
            }
        }
    }else{ // sparse
        // K k=0, j=0;
        const size_t RStart = 3+CompCCSFormat[1];
        for(K k=0, j=0; j< CompCCSFormat[1]; j++){
            for(; k< CompCCSFormat[3+j]; k++){
                iRow[wp] = CompCCSFormat[RStart+k];
                jCol[wp] = j;
                ++wp;
            }
        }
    }// if(CompCCSFormat[2]==1){ //dense
}

template <typename Derived, typename T, typename K>
void compCCS_fillDense(const T CompCCSFormat[], const K numbers[] , Eigen::DenseBase<Derived>& target)
{
    T nnz = compCCS_nnz(CompCCSFormat);
    for(int i = 0; i < nnz; i++) { std::cout << numbers[i] <<", ";}
    std::cout <<std::endl;
    size_t rp = 0; // write ptr
    if(CompCCSFormat[2]==1){ //dense
        for(T i = 0; i< CompCCSFormat[0]; i++){ // row first arrangement
            for(T j = 0; j< CompCCSFormat[1]; j++){
                target(i,j) = numbers[rp];
                ++rp;
            }
        }
    }else{ // sparse
        // K k=0, j=0;
        const size_t RStart = 3+CompCCSFormat[1];
        for(T k=0, j=0; j< CompCCSFormat[1]; j++){
            for(; k< CompCCSFormat[3+j]; k++){
                std::cout << "i,j:\t"  << CompCCSFormat[RStart+k] << "\t" <<j << ", val:\t" << numbers[rp] << std::endl;
                target(CompCCSFormat[RStart+k],j) = numbers[rp];
                ++rp;
            }
        }
    }// if(CompCCSFormat[2]==1){ //dense
}


template <typename T>
class SimpleArrayPtr{
private:
    T* ptr;
    const size_t _n;
    SimpleArrayPtr& operator= (const SimpleArrayPtr& a) = delete;
    SimpleArrayPtr(const SimpleArrayPtr& a) = delete;
public:
    SimpleArrayPtr(size_t N):_n(N){
        ptr = !N? nullptr : new T[N];
    }
    ~SimpleArrayPtr(){
        delete []ptr;
    }
    const size_t n() const{return _n;}
    T* p() {return ptr;}
    operator T* () {return ptr;}

};