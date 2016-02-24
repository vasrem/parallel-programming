#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


#define bufSize 1024
#define INF 999999

float **A;		// input Table
float tsize=0;


void print();
void make_table();
void run();

int main(){

	make_table();
	print();
	run();
	return 0;
	
}
/*	run()
 *	-----
 *	Runs Floys Floyd-Warshall's Algorithm
 */
void run(){
	int i , j , k ;

	for ( i = 0 ; i < tsize ; i++ ){
		for( j = 0 ; j < tsize ; j++ ){
			for( k = 0 ; k < tsize ; k++ ){
				if( A[i][j] > A[i][k] + A[k][j] ){
					A[i][j] = A[i][k] + A[k][j];
				}				
			}
		}
	}
	printf("\nDone!");
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
	tsize = atof(buf);
	
	// Alloc the table
	A =malloc (tsize*sizeof(float *));
	for(i=0;i<tsize;i++){
		A[i]=malloc(tsize*sizeof(float));
	}
	i=0;
	
	// Fill the table
	while(fgets(buf,sizeof(buf),fp)!=NULL){
		for(j = 0 ; j < tsize ; j++ ){
			A[i][j]=atof(&buf[16*j]);
		}
		i++;
	}

	fclose(fp);
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
			if(isinf(A[i][j])){
				printf("%f\t\t",A[i][j]);
			}else{
				printf("%f\t",A[i][j]);
			}
		}
		printf("\n-------------------------\n");
	}

}
