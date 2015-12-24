/*
bitonic.c 

This file contains two different implementations of the bitonic sort
recursive  version :  rec
imperative version :  impBitonicSort() 


The bitonic sort is also known as Batcher Sort. 
For a reference of the algorithm, see the article titled 
Sorting networks and their applications by K. E. Batcher in 1968 


The following codes take references to the codes avaiable at 

http://www.cag.lcs.mit.edu/streamit/results/bitonic/code/c/bitonic.c

http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm

http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/bitonicen.htm 
*/

/* 
------- ---------------------- 
Nikos Pitsianis, Duke CS 
-----------------------------
*/

/* Vasilis Remmas */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <cilk/cilk.h>

struct timeval startwtime, endwtime;
double seq_time;



int N;          // data array size
int *a;         // data array to be sorted
int T;          // number of threads
int t=0;          // number of active threads
int tid=0;

pthread_t *threads;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct{
  int id;
  int lo;
  int cnt;
  int dir;
}thread_data;

const int ASCENDING  = 1;
const int DESCENDING = 0;


void init(void);
void print(void);
void sort(void);
void test(void);
inline void exchange(int i, int j);
void compare(int i, int j, int dir);
void bitonicMerge(int lo, int cnt, int dir);
void recBitonicSort(int lo, int cnt, int dir);
void impBitonicSort(void);
// pthreads
void *parRecBitonicSort(void* arg);
void parSort(void);
void *parBitonicMerge(void* arg);
void *parBCRecBitonicSort(void* arg);
void parBCSort(void);
void *parBCBitonicMerge(void* arg);
// OpenMP
void parImpBitonicSort(void);
// Cilk Plus
void cilkBitonicMerge(int lo, int cnt, int dir);
void cilkRecBitonicSort(int lo, int cnt, int dir);
void cilkSort(void);
void cilkImpBitonicSort(void);


// qsort functions
int desc( const void *a, const void *b ){
  int* arg1 = (int *)a;
  int* arg2 = (int *)b;
  if( *arg1 > *arg2 ) return -1;
  else if( *arg1 == *arg2 ) return 0;
  return 1;
}
int asc( const void *a, const void *b ){
  int* arg1 = (int *)a;
  int* arg2 = (int *)b;
  if( *arg1 < *arg2 ) return -1;
  else if( *arg1 == *arg2 ) return 0;
  return 1;
}

/** the main program **/ 
int main(int argc, char **argv) {

  if (argc<3)
  {
    if (argc != 2) {
      printf("Usage: %s q\n  where n=2^q is problem size (power of two)\n", 
        argv[0]);
      exit(1);
    }
    if (argc != 3) {
      printf("Usage: %s %s p\n  where 2^p is number of threads (power of two)\n", 
        argv[0],argv[1]);
      exit(1);
    }
  }

  N = 1<<atoi(argv[1]);
  T = 1<<atoi(argv[2]);
  printf("N=%d,T=%d\n",N,T);
  a = (int *) malloc(N * sizeof(int));

  threads = (pthread_t *) malloc(20000 * sizeof(pthread_t));

  init();
  gettimeofday (&startwtime, NULL);
  qsort(a,N,sizeof(int),asc);
  gettimeofday (&endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
    + endwtime.tv_sec - startwtime.tv_sec);

  printf("Quicksort wall clock time = %f\n", seq_time);
  test();

  init();
  gettimeofday (&startwtime, NULL);
  sort();
  gettimeofday (&endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
    + endwtime.tv_sec - startwtime.tv_sec);

  printf("Recursive wall clock time = %f\n", seq_time);

  test();

  init();
  gettimeofday (&startwtime, NULL);
  parSort();
  gettimeofday (&endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
    + endwtime.tv_sec - startwtime.tv_sec);

  printf("Parallel(pthreads) w/o BC Recursive wall clock time = %f\n", seq_time);
  test();

  init();
  gettimeofday (&startwtime, NULL);
  parBCSort();
  gettimeofday (&endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
    + endwtime.tv_sec - startwtime.tv_sec);

  printf("Parallel(pthreads) w/ BC Recursive wall clock time = %f\n", seq_time);
  test();

  init();
  
  gettimeofday (&startwtime, NULL);
  cilkSort();
  gettimeofday (&endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
    + endwtime.tv_sec - startwtime.tv_sec);

  printf("Parallel(CilkPlus) Recursive wall clock time = %f\n", seq_time);
  test();

  init();

  gettimeofday (&startwtime, NULL);
  impBitonicSort();
  gettimeofday (&endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
    + endwtime.tv_sec - startwtime.tv_sec);

  printf("Imperative wall clock time = %f\n", seq_time);

  test();

  init();

  gettimeofday (&startwtime, NULL);
  parImpBitonicSort();
  gettimeofday (&endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
    + endwtime.tv_sec - startwtime.tv_sec);

  printf("Parallel(OpenMP) Imperative wall clock time = %f\n", seq_time);

  test();

  init();

  gettimeofday (&startwtime, NULL);
  cilkImpBitonicSort();
  gettimeofday (&endwtime, NULL);

  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
    + endwtime.tv_sec - startwtime.tv_sec);

  printf("Parallel(CilkPlus) Imperative wall clock time = %f\n", seq_time);

  test();

  

// print();
}

/** -------------- SUB-PROCEDURES  ----------------- **/ 

/** procedure test() : verify sort results **/
void test() {
  int pass = 1;
  int i;
  for (i = 1; i < N; i++) {
    pass &= (a[i-1] <= a[i]);
  }

  printf(" TEST %s\n",(pass) ? "PASSed" : "FAILed");
}


/** procedure init() : initialize array "a" with data **/
void init() {
  int i;
  for (i = 0; i < N; i++) {
a[i] = rand() % N; // (N - i);
}
}

/** procedure  print() : print array elements **/
void print() {
  int i;
  for (i = 0; i < N; i++) {
    printf("%d\n", a[i]);
  }
  printf("\n");
}


/** INLINE procedure exchange() : pair swap **/
inline void exchange(int i, int j) {
  int t;
  t = a[i];
  a[i] = a[j];
  a[j] = t;
}



/** procedure compare() 
The parameter dir indicates the sorting direction, ASCENDING 
or DESCENDING; if (a[i] > a[j]) agrees with the direction, 
then a[i] and a[j] are interchanged.
**/
inline void compare(int i, int j, int dir) {
  if (dir==(a[i]>a[j])) 
    exchange(i,j);
}




/** Procedure bitonicMerge() 
It recursively sorts a bitonic sequence in ascending order, 
if dir = ASCENDING, and in descending order otherwise. 
The sequence to be sorted starts at index position lo,
the parameter cbt is the number of elements to be sorted. 
**/
void bitonicMerge(int lo, int cnt, int dir) {
  if (cnt>1) {
    int k=cnt/2;
    int i;
    for (i=lo; i<lo+k; i++)
      compare(i, i+k, dir);
    bitonicMerge(lo, k, dir);
    bitonicMerge(lo+k, k, dir);
  }
}



/** function recBitonicSort() 
first produces a bitonic sequence by recursively sorting 
its two halves in opposite sorting orders, and then
calls bitonicMerge to make them in the same order 
**/
void recBitonicSort(int lo, int cnt, int dir) {
  if (cnt>1) {
    int k=cnt/2;
    recBitonicSort(lo, k, ASCENDING);
    recBitonicSort(lo+k, k, DESCENDING);
    bitonicMerge(lo, cnt, dir);
  }
}





/** function sort() 
Caller of recBitonicSort for sorting the entire array of length N 
in ASCENDING order
**/
void sort() {
  recBitonicSort(0, N, ASCENDING);
}





/*
imperative version of bitonic sort
*/
void impBitonicSort() {

  int i,j,k;

  for (k=2; k<=N; k=2*k) {
    for (j=k>>1; j>0; j=j>>1) {
      for (i=0; i<N; i++) {
        int ij=i^j;
        if ((ij)>i) {
          if ((i&k)==0 && a[i] > a[ij]) 
            exchange(i,ij);
          if ((i&k)!=0 && a[i] < a[ij])
            exchange(i,ij);
        }
      }
    }
  }
}


/* ---------------------------------------- */
/* -----------                 ------------ */
/* ----------- P T H R E A D S ------------ */
/* -----------                 ------------ */
/* ---------------------------------------- */

void *parBitonicMerge(void* arg){
  thread_data data1,data2;
  int rc;
  int lo = ((thread_data *) arg)->lo;
  int cnt = ((thread_data *) arg)->cnt;
  int dir = ((thread_data *) arg)->dir;
  int i;
  int k=cnt/2;
  if(cnt>1){
    for(i=lo;i<lo+k;++i){
      compare(i,i+k,dir);
    }
    pthread_mutex_lock(&mutex);
    int temp = t;
    if(temp<T-1){
      t+=2;
    }
    pthread_mutex_unlock(&mutex);
    if(temp>=T-1){
      bitonicMerge(lo,k,dir);
      bitonicMerge(lo+k,k,dir);
    }else{
      pthread_attr_t attr;
      pthread_attr_init(&attr);
      pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);
      pthread_mutex_lock(&mutex);
      tid+=1;
      data1.id = tid;
      // printf("CREATED merge #%d with id #%d\n",t,tid);
      pthread_mutex_unlock(&mutex);
      data1.lo = lo;
      data1.cnt = k;
      data1.dir = dir;
      rc = pthread_create(&threads[data1.id],&attr,parBitonicMerge,&data1);

      pthread_mutex_lock(&mutex);
      tid+=1;
      data2.id = tid;
      // printf("CREATED merge #%d with id #%d\n",t,tid);
      pthread_mutex_unlock(&mutex);
      data2.lo = lo+k;
      data2.cnt = k;
      data2.dir = dir;
      rc = pthread_create(&threads[data2.id],&attr,parBitonicMerge,&data2);
      rc = pthread_join(threads[data1.id],NULL);
      rc = pthread_join(threads[data2.id],NULL);
      pthread_attr_destroy(&attr);

      pthread_mutex_lock(&mutex);
      t-=2;
      pthread_mutex_unlock(&mutex);

    }
  }
  return 0;
}



void *parRecBitonicSort(void* arg){
  thread_data data1,data2,data3;
  int rc;
  int id = ((thread_data *) arg)->id;
  int lo = ((thread_data *) arg)->lo;
  int cnt = ((thread_data *) arg)->cnt;
  int dir = ((thread_data *) arg)->dir;
  int k = cnt/2;
  if(cnt>1){
    pthread_mutex_lock(&mutex);
    int temp = t;
    if(temp<T-1){
      t+=2;
    }
    pthread_mutex_unlock(&mutex);
    if(temp>=T-1){
      qsort(a+lo,k,sizeof(int),asc);
      qsort(a+lo+k,k,sizeof(int),desc);
    }else{
      pthread_attr_t attr;
      pthread_attr_init(&attr);
      pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);
      pthread_mutex_lock(&mutex);
      tid+=1;
      data1.id = tid;
      // printf("CREATED sort #%d with id #%d\n",t,tid);
      pthread_mutex_unlock(&mutex);
      data1.lo = lo;
      data1.cnt = k;
      data1.dir = ASCENDING;
      rc = pthread_create(&threads[data1.id],&attr,parRecBitonicSort,&data1);
      pthread_mutex_lock(&mutex);
      tid+=1;
      data2.id = tid;
      // printf("CREATED sort #%d with id #%d\n",t,tid);
      pthread_mutex_unlock(&mutex);
      data2.lo = lo+k;
      data2.cnt = k;
      data2.dir = DESCENDING;
      rc = pthread_create(&threads[data2.id],&attr,parRecBitonicSort,&data2);
      rc = pthread_join(threads[data1.id],NULL);
      rc = pthread_join(threads[data2.id],NULL);
      pthread_attr_destroy(&attr);

      pthread_mutex_lock(&mutex);
      t-=2;
      pthread_mutex_unlock(&mutex);
    } 
    data3.lo = lo;
    data3.cnt = cnt;
    data3.dir = dir;
    parBitonicMerge(&data3);
  }
  return 0;
}

void parSort(){
  thread_data data;
  data.id = tid;
  data.lo = 0;
  data.cnt = N;
  data.dir = ASCENDING;
  parRecBitonicSort(&data);
}

void *parBCBitonicMerge(void* arg){
  thread_data data1,data2;
  int rc;
  int lo = ((thread_data *) arg)->lo;
  int cnt = ((thread_data *) arg)->cnt;
  int dir = ((thread_data *) arg)->dir;
  int i;
  int k=cnt/2;
  if(cnt>1){
    for(i=lo;i<lo+k;++i){
      compare(i,i+k,dir);
    }
    pthread_mutex_lock(&mutex);
    int temp = t;
    if(temp<T-1){
      t+=2;
    }
    pthread_mutex_unlock(&mutex);
    if(temp>=T-1){
      bitonicMerge(lo,k,dir);
      bitonicMerge(lo+k,k,dir);
    }else{
      pthread_attr_t attr;
      pthread_attr_init(&attr);
      pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);
      pthread_mutex_lock(&mutex);
      tid+=1;
      data1.id = tid;
      // printf("CREATED merge #%d with id #%d\n",t,tid);
      pthread_mutex_unlock(&mutex);
      data1.lo = lo;
      data1.cnt = k;
      data1.dir = dir;
      rc = pthread_create(&threads[data1.id],&attr,parBCBitonicMerge,&data1);

      pthread_mutex_lock(&mutex);
      tid+=1;
      data2.id = tid;
      // printf("CREATED merge #%d with id #%d\n",t,tid);
      pthread_mutex_unlock(&mutex);
      data2.lo = lo+k;
      data2.cnt = k;
      data2.dir = dir;
      rc = pthread_create(&threads[data2.id],&attr,parBCBitonicMerge,&data2);
      rc = pthread_join(threads[data1.id],NULL);
      rc = pthread_join(threads[data2.id],NULL);
      pthread_attr_destroy(&attr);

      pthread_mutex_lock(&mutex);
      t-=2;
      pthread_mutex_unlock(&mutex);

    }
  }
  return 0;
}



void *parBCRecBitonicSort(void* arg){
  thread_data data1,data2,data3;
  int rc;
  int id = ((thread_data *) arg)->id;
  int lo = ((thread_data *) arg)->lo;
  int cnt = ((thread_data *) arg)->cnt;
  int dir = ((thread_data *) arg)->dir;
  int k = cnt/2;
  if(cnt>1){
    pthread_mutex_lock(&mutex);
    int temp = t;
    if(temp<T-1){
      if(k>(1<<15)){
        t+=2;
      }
    }
    pthread_mutex_unlock(&mutex);
    if(temp>=T-1 || k<=(1<<15)){
      qsort(a+lo,k,sizeof(int),asc);
      qsort(a+lo+k,k,sizeof(int),desc);
    }else{
      pthread_attr_t attr;
      pthread_attr_init(&attr);
      pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);
      pthread_mutex_lock(&mutex);
      tid+=1;
      data1.id = tid;
      // printf("CREATED sort #%d with id #%d\n",t,tid);
      pthread_mutex_unlock(&mutex);
      data1.lo = lo;
      data1.cnt = k;
      data1.dir = ASCENDING;
      rc = pthread_create(&threads[data1.id],&attr,parBCRecBitonicSort,&data1);
      pthread_mutex_lock(&mutex);
      tid+=1;
      data2.id = tid;
      // printf("CREATED sort #%d with id #%d\n",t,tid);
      pthread_mutex_unlock(&mutex);
      data2.lo = lo+k;
      data2.cnt = k;
      data2.dir = DESCENDING;
      rc = pthread_create(&threads[data2.id],&attr,parBCRecBitonicSort,&data2);
      rc = pthread_join(threads[data1.id],NULL);
      rc = pthread_join(threads[data2.id],NULL);
      pthread_attr_destroy(&attr);

      pthread_mutex_lock(&mutex);
      t-=2;
      pthread_mutex_unlock(&mutex);
    } 
    data3.lo = lo;
    data3.cnt = cnt;
    data3.dir = dir;
    parBitonicMerge(&data3);
  }
  return 0;
}

void parBCSort(){
  thread_data data;
  data.id = tid;
  data.lo = 0;
  data.cnt = N;
  data.dir = ASCENDING;
  parBCRecBitonicSort(&data);
}

/* ------------------------------------ */
/* -----------             ------------ */
/* ----------- O P E N M P ------------ */
/* -----------             ------------ */
/* ------------------------------------ */

void parImpBitonicSort() {
  omp_set_num_threads(T);
  int i,j,k;

  for (k=2; k<=N; k=2*k) {
    for (j=k>>1; j>0; j=j>>1) {
      #pragma omp parallel for
        for (i=0; i<N; i++) {
          int ij=i^j;
          if ((ij)>i) {
            if ((i&k)==0 && a[i] > a[ij]) 
              exchange(i,ij);
            if ((i&k)!=0 && a[i] < a[ij])
              exchange(i,ij);
          }
        }
    }
  }
}


/* -------------------------------- */
/* -----------         ------------ */
/* ----------- C I L K ------------ */
/* -----------         ------------ */
/* -------------------------------- */
void cilkBitonicMerge(int lo, int cnt, int dir) {
  if (cnt>1) {
    int k=cnt/2;
    int i;
    for (i=lo; i<lo+k; i++)
      compare(i, i+k, dir);
    _Cilk_spawn cilkBitonicMerge(lo, k, dir);
    _Cilk_spawn cilkBitonicMerge(lo+k, k, dir);
    _Cilk_sync;
  }
}

void cilkRecBitonicSort(int lo, int cnt, int dir) {
  if (cnt>1) {
    int k=cnt/2;
    _Cilk_spawn cilkRecBitonicSort(lo, k, ASCENDING);
    _Cilk_spawn cilkRecBitonicSort(lo+k, k, DESCENDING);
    _Cilk_sync;
    cilkBitonicMerge(lo, cnt, dir);
  }
}
void cilkSort() {
  cilkRecBitonicSort(0, N, ASCENDING);
}


void cilkImpBitonicSort() {
  __cilkrts_set_param("nworkers", T);

  int i,j,k;

  for (k=2; k<=N; k=2*k) {
    for (j=k>>1; j>0; j=j>>1) {
      cilk_for (i=0; i<N; i++) {
        int ij=i^j;
        if ((ij)>i) {
          if ((i&k)==0 && a[i] > a[ij]) 
            exchange(i,ij);
          if ((i&k)!=0 && a[i] < a[ij])
            exchange(i,ij);
        }
      }
    }
  }
}

