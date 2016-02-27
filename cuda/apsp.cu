#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#define bufSize 700000 


struct timeval startwtime,endwtime;

float *h_a;			// Table at host
float *d_a;			// Table at device
int tsize=0;		// number of rows or columns
size_t size = 0 ;	// size of table( tsize* tsize * sizeof(float*))
float* test;

void print(float *);
void make_table();
void serial();
void check();
void copytables();

__global__ void Kernel1(float *A,int N,int k){

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	//printf("Hello from %d %d \n",threadIdx.x,threadIdx.y);
	if ( A[i*N+j] > A[i*N+k] + A[k*N+j] ){
		A[i*N+j] = A[i*N+k] + A[k*N+j];
	}
}

__global__ void Kernel2(float *A,int N,int k){
	
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	__shared__ float k_k1,k1_k;
	/*	example:
	 *	if we have to from D -> F throw k and then k+1 we have to do:
	 *	DkF check
	 *  D(k+1)F check
	 *	Dk(k+1)F check
	 *	D(k+1)kF check
	 *	the min of these is the min dist.
	 */
	if(threadIdx.x==0 && threadIdx.y==0){
		k_k1=A[k*N+(k+1)];
		k1_k=A[(k+1)*N+k];
	}
	float x,y,asked,xn,yn;

	asked=A[i*N+j];
	
	x=A[k*N+j];
	y=A[i*N+k];

	// DkF
	if(asked>x+y){
		asked=x+y;
	}

	xn=A[i*N+(k+1)];
	yn=A[(k+1)*N+j];

	__syncthreads();
	
	//	D(k+1)
	if(xn>y+k_k1){
		xn=y+k_k1;
	}
	//	(k+1)F
	if(yn>x+k1_k){
		yn=x+k1_k;
	}
	//	D(k+1)F or D(k+1)kF or Dk(k+1)F
	if(asked>xn+yn){
		asked=xn+yn;
	}
	//	min dist
	A[i*N+j]=asked;
}

int main(){

	make_table();
	gettimeofday(&startwtime,NULL);

	serial();

	gettimeofday(&endwtime,NULL);
	printf("Serial time : %lf\n",	(double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec));

	copytables();
	free(h_a);

	// ----------------------------
	//           Kernel 1
	// ----------------------------
	
	make_table();
	gettimeofday(&startwtime,NULL);

	// Alloc device table
	cudaMalloc((void **)&d_a,size);

	// Transfer table to device
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);	

	// Define dimensions	
	int threads = 32;
	dim3 dimBlock(threads,threads);
	dim3 dimGrid(tsize/dimBlock.x,tsize/dimBlock.y);	

	// Do the math
	int k = 0;
	for ( k = 0 ; k < tsize ; k++){
		Kernel1<<<dimGrid,dimBlock>>>(d_a,tsize,k);
		cudaDeviceSynchronize();	
	}
	
	// Transfer table to host

	cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);

	gettimeofday(&endwtime,NULL);
	printf("Kernel 1 time : %lf\n",	(double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec));
	
	// Checks the result
	check();
	
	// Free device and host memory
	cudaFree(d_a);
	free(h_a);


	// ----------------------------
	//           Kernel 2
	// ----------------------------
	
	make_table();
	gettimeofday(&startwtime,NULL);

	// Alloc device table
	cudaMalloc((void **)&d_a,size);

	// Transfer table to device
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);	

	// Do the math
	for ( k = 0 ; k < tsize ; k+=2){
		Kernel2<<<dimGrid,dimBlock>>>(d_a,tsize,k);
		cudaDeviceSynchronize();	
	}
	
	// Transfer table to host

	cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);

	gettimeofday(&endwtime,NULL);
	printf("Kernel 2 time : %lf\n",	(double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec));


	// Checks the result
	check();
	
	// Free device and host memory
	cudaFree(d_a);
	free(h_a);

	return 0;
	
}


/*	serial()
 *	-----
 *	Runs serial Floys Floyd-Warshall's Algorithm
 */
void serial(){
	int i , j , k ;

	for ( k = 0 ; k < tsize ; k++ ){
		for( i = 0 ; i < tsize ; i++ ){
			for( j = 0 ; j < tsize ; j++ ){

				if( h_a[i*tsize+j] > h_a[i*tsize+k] + h_a[k*tsize+j] ){
					h_a[i*tsize+j] = h_a[i*tsize+k] + h_a[k*tsize+j];
				}				
			}
		}
	}
	/*printf("\nDone Serial.\n");*/
}


/*	make_table()
 *	------------
 *	Gets the input table at table A
 *	input file has at first line size of table 
 *	and at the other lines the content.
 */
void make_table(){
	FILE * fp;
	char buf[bufSize];
	int i = 0 ;
	int j = 0 ;
 	
	fp = fopen("input.txt","r");
	
	// Read size of table
	fgets(buf,sizeof(buf),fp);
	tsize =(int) atof(buf);
	size = tsize*tsize*sizeof(float);	
	// Alloc the table at host
	h_a =(float *) malloc (size);
	// Fill the table
	while(fgets(buf,sizeof(buf),fp)!=NULL){
		for(j = 0 ; j < tsize ; j++ ){
			if(isinf(atof(&buf[16*j]))){
				h_a[i*tsize+j]=999999;
			}else{
				h_a[i*tsize+j]=atof(&buf[16*j]);
			}
		}
		i++;
	}

	fclose(fp);
	/*printf("\nDone making table.\n");*/
}



/*	print()
 *	-------
 *	Prints the table A	
 */
void print(float *a){
	int i = 0;
	int j = 0;
	for(i=0;i<tsize;i++){
		for(j=0;j<tsize;j++){
			if(isinf(h_a[i*tsize+j])){
				printf("%f\t\t",a[i*tsize+j]);
			}else{
				printf("%f\t",a[i*tsize+j]);
			}
		}
		printf("\n-------------------------\n");
	}

}

/*	check()
 *	-------
 *	Checks if the produced table of Kernel is the same
 *	as the serial's table.
 */
void check(){
	int i = 0;
	int j = 0;
	for(i=0;i<tsize;i++){
		for(j=0;j<tsize;j++){
			if(fabs(h_a[i*tsize+j]-test[i*tsize+j])>1e-6){
				/*printf("Error at [%d][%d] -> %f %f.\n",i,j,h_a[i*tsize+j],test[i*tsize+j]);*/
				printf("Fail!\n");
				exit(1);
			}
		}
	}
	printf("Success!\n");
}


/*	copytables()
 *	------------
 *	Copys the serial's table to test table.
 *	Test table is used at check().
 */
void copytables(){
	int i = 0;
	test=(float *) malloc (size);	
	for(i=0;i<tsize*tsize;i++){
		test[i]=h_a[i];
	}
}
