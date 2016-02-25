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


void print();
void make_table();
void serial();

__global__ void Kernel1(float *A,int N,int k){

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if ( A[i*N+j] > A[i*N+k] + A[k*N+j] ){
		A[i*N+j] = A[i*N+k] + A[k*N+j];
	}
}

int main(){

	make_table();
	/*print();*/

	gettimeofday(&startwtime,NULL);

	serial();
	
	gettimeofday(&endwtime,NULL);
	printf("Serial time : %lf\n",	(double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec));

	// ----------------------------
	//           Kernel 1
	// ----------------------------
	
	make_table();
	gettimeofday(&startwtime,NULL);

	// Alloc device table
	cudaMalloc(&d_a,size);

	// Transfer table to device
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);	

	// Define dimensions	
	int threads = tsize;	
	dim3 dimBlock(threads,threads);
	dim3 dimGrid(tsize/dimBlock.x,tsize/dimBlock.y);	

	// Do the math
	int k = 0;
	for ( k = 0 ; k < tsize ; k++){
		Kernel1<<<dimGrid,dimBlock>>>(d_a,tsize,k);
		cudaThreadSynchronize();
	}
	
	// Transfer table to host

	cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);

	// Free device and host memory
	
	cudaFree(d_a);
	free(h_a);

	gettimeofday(&endwtime,NULL);
	printf("Kernel 1 time : %lf\n",	(double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec));

	return 0;
	
}


/*	serial()
 *	-----
 *	Runs serial Floys Floyd-Warshall's Algorithm
 */
void serial(){
	int i , j , k ;

	for ( i = 0 ; i < tsize ; i++ ){
		for( j = 0 ; j < tsize ; j++ ){
			for( k = 0 ; k < tsize ; k++ ){
				if( h_a[i*tsize+j] > h_a[i*tsize+k] + h_a[k*tsize+j] ){
					h_a[i*tsize+j] = h_a[i*tsize+k] + h_a[k*tsize+j];
				}				
			}
		}
	}
	printf("\nDone Serial.\n");
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
			h_a[i*tsize+j]=atof(&buf[16*j]);
		}
		i++;
	}

	fclose(fp);
	printf("\nDone making table.\n");
}



/*	print()
 *	-------
 *	Prints the table A	
 */
void print(){
	int i = 0;
	int j = 0;
	for(i=0;i<tsize;i++){
		for(j=0;j<tsize;j++){
			if(isinf(h_a[i*tsize+j])){
				printf("%f\t\t",h_a[i*tsize+j]);
			}else{
				printf("%f\t",h_a[i*tsize+j]);
			}
		}
		printf("\n-------------------------\n");
	}

}



