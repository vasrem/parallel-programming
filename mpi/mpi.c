
/* Vasilis Remmas */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include "mpi.h"
 
int Nc;	// C size 	[21:25]
int Nq;	// Q size 	[21:25]
int gs; // n x m x k [12:16]
float n;	// x axis size per process
float m;	// y axis size per process
float k;	// z axis size per process
float no; // x axis size
float mo; // y axis size
float ko; // z axis size
int P;	// number of processes [0:7]
int L;	// Used for table allocation and make_grid()

double *C,*Co,*Ci; // C , Coutput , Cinput
double *Q,*Qo,*Qi; // Q , Qoutput , Qinput


/* ic is the real size of C ( Not zero )
* jc is the real size of Co ( Not zero )
* iq 			>>		  Q
* jq			>>		  Qo
*/
int ic=0,jc=0;
int iq=0,jq=0;

//Bounds 
float nbl=0;	// Low X bound
float mbl=0;	// Low Y bound
float kbl=0;	// Low Z bound
float nbh=0;	// High X bound
float mbh=0;	// High Y bound
float kbh=0;	// High Z bound


void make_grid(int);
void check_inc_C();
void check_inc_Q();
void print_table(double *,int,int,int);
void quicksort(double *,int,int,int);


int main(int argc, char **argv){
	int numtasks, rank; 
	int temp,i;
	float buf[6];
	MPI_Status *stats;
	stats = (MPI_Status *) malloc(128*sizeof(MPI_Status));
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if( argc < 5 ){
		if( argc == 1 ){
			printf("Usage: %s Nc\n  where Nc is C size (power of two)\n", 
        	argv[0]);
        	exit(1);
		}
		if( argc == 2 ){
			printf("Usage: %s %s Nq\n  where Nq is Q size (power of two)\n", 
        	argv[0],argv[1]);
        	exit(1);
		}
		if( argc == 3 ){
			printf("Usage: %s %s %s gs\n  where gs= n x m x k is grid size (power of two)\n", 
        	argv[0],argv[1],argv[2]);
        	exit(1);
		}
		if( argc == 4 ){
			printf("Usage: %s %s %s %s P\n  where P is number of processes (power of two)\n", 
        	argv[0],argv[1],argv[2],argv[3]);
        	exit(1);
		}
	}
	Nc=1<<atoi(argv[1]);
	Nq=1<<atoi(argv[2]);
	gs=1<<atoi(argv[3]);
	P=1<<atoi(argv[4]);

	Ci=(double **) malloc(Nc*sizeof(double *));
	Qi=(double **) malloc(Nq*sizeof(double *));

	for(i=0;i<Nc;i++){
		Ci[i]=(double *) malloc(3*sizeof(double));
	}

	for(i=0;i<Nq;i++){
		Qi[i]=(double *) malloc(3*sizeof(double));
	}

	temp = log( gs ) / log(2);
	for(i=0;i< temp ; i++){
		ko++;
		i++;
		if(i>=temp){
			break;
		}
		mo++;
		i++;
		if(i>=temp){
			break;
		}
		no++;
	}
	no=1<<(int)no;
	mo=1<<(int)mo;
	ko=1<<(int)ko;
	temp = log( gs / P ) / log(2);
	for(i=0;i< temp ; i++){
		k++;
		i++;
		if(i>=temp){
			break;
		}
		m++;
		i++;
		if(i>=temp){
			break;
		}
		n++;
	}
	n=1<<(int)n;
	m=1<<(int)m;
	k=1<<(int)k;
	printf("%d %d %d %d\n",Nc,Nq,gs,P);
	printf("n=%f m=%f k=%f\n",n,m,k);
	printf("no=%f mo=%f ko=%f\n",no,mo,ko);
	if(rank==0){
		// printf("Number of active processes is %d\n",numtasks);
		int i=1;
		double s1,s2,s3;
		//Give right bounds
		for(s1=0;s1<1;s1+=n/no){
			buf[0]=s1;
			buf[3]=s1+(n/no);
			for(s2=0;s2<1;s2+=m/mo){
				buf[1]=s2;
				buf[4]=s2+(m/mo);
				for(s3=0;s3<1;s3+=k/ko){
					buf[2]=s3;
					buf[5]=s3+(k/ko);
					printf("%f %f %f %f %f %f\n",buf[0],buf[1],buf[2],buf[3],buf[4],buf[5]);
					MPI_Send(&buf[0],6,MPI_FLOAT,i,i,MPI_COMM_WORLD);
					i++;
				}	
			}
		}

	}else{
		MPI_Recv (&buf[0],6,MPI_FLOAT,0,rank,MPI_COMM_WORLD,&stats[rank-1]);
		nbl=buf[0];	// Low X bound
		mbl=buf[1];	// Low Y bound
		kbl=buf[2];	// Low Z bound
		nbh=buf[3];	// High X bound
		mbh=buf[4];	// High Y bound
		kbh=buf[5];	// High Z bound
		// printf("Process #%d\nnl=%f ml=%f kl=%f nh=%f mh=%f kh=%f\n",rank,nbl,mbl,kbl,nbh,mbh,kbh);
		make_grid(rank);
		for(i=0;i<jc;i++){
			MPI_Allgather (&Co[i][0],3,MPI_DOUBLE,&Ci[i][0],3*(numtasks-1),MPI_DOUBLE,MPI_COMM_WORLD);
		}
		printf("ARGH");

	}


	// make_grid();
	


	// // --------
	// // PROXEIRO
	// // ---------
	// L=Nc/P;
	// int s;
	// i=0;
	// Ci = (double **) malloc(L*sizeof(double*));
	// // Temp table of rand numbers
	// double *d;
	// d =(double *) malloc(3*sizeof(double));
	// // srand setup 
	// struct timeval time; 
	// gettimeofday(&time,NULL);
	// srand((time.tv_sec * 1500) + (time.tv_usec / 1500));

	// for( s = 0 ; s < L ; s++ ){
	// 	Ci[s] = (double *) malloc(4*sizeof(double));

	// 	d[0] = (double) rand()/RAND_MAX;
	// 	d[1] = (double) rand()/RAND_MAX;
	// 	d[2] = (double) rand()/RAND_MAX;
	// 	// printf("C:%0.10f\t%0.10f\t%0.10f\t\n",d[0],d[1],d[2]);

	// // if the number is ok for this process
	// 	if( d[0] < 1 && d[1] < 1 && d[2] < 1 && d[0] >= 0 && d[1] >= 0 && d[2] >= 0 ){
	// 		Ci[i][0]=d[0];
	// 		Ci[i][1]=d[1];
	// 		Ci[i][2]=d[2];
	// 		i++;
	// 	}
	// }
	// printf("---Table Ci---\n");
	// print_table(Ci,i,3);
	// check_inc_C();

	// // --------
	// // PROXEIRO
	// // ---------
	MPI_Finalize();

}

void check_inc_C(){
	int s;
	int A=0;
	double a;
	L = Nc / P;
	printf("nl=%f ml=%f kl=%f nh=%f mh=%f kh=%f\n",nbl,mbl,kbl,nbh,mbh,kbh);
	for(s=0;s<L;s++){
		// if end of real C break
		if(Ci[0*L+s]==0 && Ci[1*L+s]==0 && Ci[2*L+s]==0){
			break;
		}
		// if its ok for that process
		if(Ci[0*L+s] < nbh && Ci[1*L+s] < mbh && Ci[2*L+s] < kbh && Ci[0*L+s] >= nbl && Ci[1*L+s] >= mbl && Ci[2*L+s] >= kbl){
			C[0*L+ic]=0;
			C[1*L+ic]=0;
			C[2*L+ic]=0;
			C[3*L+ic]=0;
	// Find x coordinate
			for(a = nbl ; a < nbh ; a = a + (nbh-nbl)/n){
				if(Ci[0*L+s]<a){
					break;
				}
				A++;
			}
			A*=100;
	// Find y coordinate
			for(a = mbl ; a < mbh ; a = a + (mbh-nbl)/m){
				if(Ci[1*L+s]<a){
					break;
				}
				A++;
			}
			A*=100;
	// Find z coordinate
			for(a = kbl ; a < kbh ; a = a + (kbh-kbl)/k){
				if(Ci[2*L+s]<a){
					break;
				}
				A++;
			}
			C[0*L+ic]=Ci[0*L+s];
			C[1*L+ic]=Ci[0*L+s];
			C[2*L+ic]=Ci[0*L+s];
			C[3*L+ic]=A;
			ic++;	
			A=0;
		}
	}
	quicksort(C,0,ic-1,L);
	printf("---Table C---\n");
	print_table(C,ic,4,L); 
}

void check_inc_Q(){
	int s;
	int A=0;
	double a;
	L = Nq / P;
	for(s=0;s<L;s++){
		// if end of real C break
		if(Qi[0*L+s]==0 && Qi[1*L+s]==0 && Qi[2*L+s]==0){
			break;
		}
		// if its ok for that process
		if(Qi[0*L+s]< nbh && Qi[1*L+s] < mbh && Qi[2*L+s] < kbh && Qi[0*L+s] >= nbl && Qi[1*L+s] >= mbl && Qi[2*L+s] >= kbl){
			Q[0*L+s]=0;
			Q[1*L+s]=0;
			Q[2*L+s]=0;
			Q[3*L+s]=0;
	// Find x coordinate
			for(a = nbl ; a < nbh ; a = a + (nbh-nbl)/n){
				if(Qi[0*L+s]<a){
					break;
				}
				A++;
			}
			A*=100;
	// Find y coordinate
			for(a = mbl ; a < mbh ; a = a + (mbh-mbl)/m){
				if(Qi[1*L+s]<a){
					break;
				}
				A++;
			}
			A*=100;
	// Find z coordinate
			for(a = kbl ; a < kbh ; a = a + (kbh-kbl)/k){
				if(Qi[2*L+s]<a){
					break;
				}
				A++;
			}
			Q[0*L+iq]=Qi[0*L+s];
			Q[1*L+iq]=Qi[1*L+s];
			Q[2*L+iq]=Qi[2*L+s];
			Q[3*L+iq]=A;
			iq++;	
			A=0;
		}
	}
	quicksort(Q,0,iq-1,L);
	printf("---Table Q---\n");
	print_table(Q,iq,4,L); 
}

void make_grid(int rank){

	int s,q;
	double a;

	/* XXYYZZ where :
	XX is x axis coordinate 
	YY is y axis coordinate
	ZZ is z axis coordinate
	1 point is high_bound_<axis> / box_axis 
	ex. 10302 means x=1 y=3 z=2 box. 
	ex. 111203 means x=11 y=12 z=03 box.
	All numbers starts from 1 and finish to variable box.
	*/
	int A=0;

// Temp table of rand numbers
	double *d;
	d =(double *) malloc(3*sizeof(double));

// srand setup 
	struct timeval time; 
	gettimeofday(&time,NULL);
	srand((time.tv_sec * 1000) + (time.tv_usec / 1000));

// Malloc tables
	
	L=Nc/P;
	C =(double *) malloc(L*4*sizeof(double));
	Co = (double *) malloc(L*3*sizeof(double));
	L=Nq/P;
	Q =(double *) malloc(L*4*sizeof(double*));
	Qo =(double *) malloc(L*3*sizeof(double*));
	

// Generate C

	L=Nc/P;
	for( s = 0 ; s < L ; s++ ){

		d[0] = (double) rand()/RAND_MAX;
		d[1] = (double) rand()/RAND_MAX;
		d[2] = (double) rand()/RAND_MAX;
		//printf("C:%0.10f\t%0.10f\t%0.10f\t\n",d[0],d[1],d[2]);

	// if the number is ok for this process
		if( d[0] < nbh && d[1] < mbh && d[2] < kbh && d[0] >= nbl && d[1] >= mbl && d[2] >= kbl ){
			C[0*L+ic]=0;
			C[1*L+ic]=0;
			C[2*L+ic]=0;
			C[3*L+ic]=0;
	// Find x coordinate
			for(a = nbl ; a < nbh ; a = a + (nbh-nbl)/n){
				if(d[0]<a){
					break;
				}
				A++;
			}
			A*=100;
	// Find y coordinate
			for(a = mbl ; a < mbh ; a = a + (mbh-mbl)/m){
				if(d[1]<a){
					break;
				}
				A++;
			}
			A*=100;
	// Find z coordinate
			for(a = kbl ; a < kbh ; a = a + (kbh-kbl)/k){
				if(d[2]<a){
					break;
				}
				A++;
			}
			C[0*L+ic]=d[0];
			C[1*L+ic]=d[1];
			C[2*L+ic]=d[2];
			C[3*L+ic]=A;
			ic++;	
			A=0;
		}else{
			Co[0*L+jc]=d[0];
			Co[1*L+jc]=d[1];
			Co[2*L+jc]=d[2];
			jc++;

		}
	}

// Generate Q

	L=Nq/P;
	for( s = 0 ; s < L ; s++ ){


		d[0] = (double) rand()/RAND_MAX;
		d[1] = (double) rand()/RAND_MAX;
		d[2] = (double) rand()/RAND_MAX;
	 	//printf("Q:%0.10f\t%0.10f\t%0.10f\t\n",d[0],d[1],d[2]);

	// if the number is ok for this process
		if( d[0] < nbh && d[1] < mbh && d[2] < kbh && d[0] >= nbl && d[1] >= mbl && d[2] >= kbl ){
			Q[0*L+iq]=0;
			Q[1*L+iq]=0;
			Q[2*L+iq]=0;
			Q[3*L+iq]=0;
			
	// Find x coordinate
			for(a = nbl ; a < nbh ; a = a + (nbh-nbl)/n){
				if(d[0]<a){
					break;
				}
				A++;
			}
			A*=100;
	// Find y coordinate
			for(a = mbl ; a < mbh ; a = a + (mbh-mbl)/m){
				if(d[1]<a){
					break;
				}
				A++;
			}
			A*=100;
	// Find z coordinate
			for(a = kbl ; a < kbh ; a = a + (kbh-kbl)/k){
				if(d[2]<a){
					break;
				}
				A++;
			}
			Q[0*L+iq]=d[0];
			Q[1*L+iq]=d[1];
			Q[2*L+iq]=d[2];
			Q[3*L+iq]=A;
			iq++;
			A=0;
		}else{
			Qo[0*L+jq]=d[0];
			Qo[1*L+jq]=d[1];
			Qo[2*L+jq]=d[2];
			jq++;
		}
	}


	quicksort(C,0,ic-1,Nc/P);
	quicksort(Q,0,iq-1,Nq/P);
	sleep(rank);
	L=Nc/P;
	printf("---Table C---\n");
	print_table(C,ic,4,L); 
	printf("---Table Co---\n");
	print_table(Co,jc,3,L);
	L=Nq/P;
	printf("---Table Q---\n");
	print_table(Q,iq,4,L); 
	printf("---Table Qo---\n");
	print_table(Qo,jq,3,L);

}

void print_table(double *t,int x,int y,int size){
	int i,j;
	for(i = 0 ; i < x ; i++){
		for(j = 0 ; j < y; j++){
			printf("[%d][%d] = %2.10f\t",i,j,t[j*size+i]);
		}
		printf("\n");
	}
}

void quicksort(double *t,int first,int last,int size){
	int pivot,j,i,k;
	double temp;

	if(first<last){
		pivot=first;
		i=first;
		j=last;

		while(i<j){
			while(t[3*size+i]<=t[3*size+pivot]&&i<last)
				i++;
			while(t[3*size+j]>t[3*size+pivot])
				j--;
			if(i<j){
				for(k=0;k<4;k++){
					temp=t[k*size+i];
					t[k*size+i]=t[k*size+j];
					t[k*size+j]=temp;
				}
			}
		}
		for(k=0;k<4;k++){
			temp=t[k*size+pivot];
			t[k*size+pivot]=t[k*size+j];
			t[k*size+j]=temp;
		}

		quicksort(t,first,j-1,size);
		quicksort(t,j+1,last,size);

	}
}
