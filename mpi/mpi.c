
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
int numtasks, rank; 

void make_grid();
void check_inc_C();
void check_inc_Q();
void print_table(double *,int,int,int);
void quicksort(double *,int,int,int);


int main(int argc, char **argv){
	int temp,i;
	float buf[6];
	MPI_Status *stats;
	stats = (MPI_Status *) malloc(128*sizeof(MPI_Status));
	MPI_Group workers;
	MPI_Comm workers_comm;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_group(MPI_COMM_WORLD, &workers);
	MPI_Group wg;
	MPI_Comm_group(MPI_COMM_WORLD, &wg);
	int *ranks;
	ranks=(int *) malloc((numtasks-1)*sizeof(int));
	for(i=0;i<numtasks;i++){
		ranks[i]=i+1;
		// printf("e%d %d\n",i,ranks[i]);
	}
	MPI_Group mg;
	MPI_Group_incl(wg, numtasks-1, ranks, &mg);
	MPI_Comm mc;
	MPI_Comm_create_group(MPI_COMM_WORLD, mg,5550, &mc);
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

	Ci=(double *) malloc((Nc/P)*3*(numtasks-1)*sizeof(double ));
	Qi=(double *) malloc((Nq/P)*3*(numtasks-1)*sizeof(double ));

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

	// Number of all boxes
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
	// Number of boxes per process
	n=1<<(int)n;
	m=1<<(int)m;
	k=1<<(int)k;
	// printf("%d %d %d %d\n",Nc,Nq,gs,P);
	// printf("n=%f m=%f k=%f\n",n,m,k);
	// printf("no=%f mo=%f ko=%f\n",no,mo,ko);
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
					// printf("%f %f %f %f %f %f\n",buf[0],buf[1],buf[2],buf[3],buf[4],buf[5]);
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
		make_grid();
		MPI_Allgather(&Co[0],(Nc/P)*3,MPI_DOUBLE,&Ci[0],(Nc/P)*3,MPI_DOUBLE,mc);	
		MPI_Barrier(mc);
		MPI_Allgather(&Qo[0],(Nq/P)*3,MPI_DOUBLE,&Qi[0],(Nq/P)*3,MPI_DOUBLE,mc);
		MPI_Barrier(mc);
		// if(rank==1){
		// 	for(i=0;i<(Nc/P)*3*(numtasks-1);i++){
		// 		printf("%f\t",Qi[i]);
		// 	}
		// }
		check_inc_C();
		check_inc_Q();

	}

	MPI_Finalize();

}

void check_inc_C(){
	int s,z=0;
	int A=0;
	double a;
	L = (Nc / P);
	for(s=0;s<=L*3;s++){
		// if its not out of limits
		if(ic>=L){
			break;
		}
		// printf("%d %d %d\n",z*L+s,(z+1)*L+s,(z+2)*L+s);
		// Jump to the next subtable when zeros
		if((Ci[z*L+s]==0 && Ci[(z+1)*L+s]==0 && Ci[(z+2)*L+s]==0) || s==L*3){
			z+=3;
			if(z<(numtasks-1)*3){
				s=-1;
				continue;
			}else{
				break;
			}
			
		}
		// if its ok for that process
		if(Ci[z*L+s] < nbh && Ci[(z+1)*L+s] < mbh && Ci[(z+2)*L+s] < kbh && Ci[z*L+s] >= nbl && Ci[(z+1)*L+s] >= mbl && Ci[(z+2)*L+s] >= kbl){
			C[0*L+ic]=0;
			C[1*L+ic]=0;
			C[2*L+ic]=0;
			C[3*L+ic]=0;
	// Find x coordinate
			for(a = nbl ; a < nbh ; a = a + (nbh-nbl)/n){
				if(Ci[z*L+s]<a){
					break;
				}
				A++;
			}
			A*=100;
	// Find y coordinate
			for(a = mbl ; a < mbh ; a = a + (mbh-nbl)/m){
				if(Ci[(z+1)*L+s]<a){
					break;
				}
				A++;
			}
			A*=100;
	// Find z coordinate
			for(a = kbl ; a < kbh ; a = a + (kbh-kbl)/k){
				if(Ci[(z+2)*L+s]<a){
					break;
				}
				A++;
			}
			C[0*L+ic]=Ci[z*L+s];
			C[1*L+ic]=Ci[(z+1)*L+s];
			C[2*L+ic]=Ci[(z+2)*L+s];
			C[3*L+ic]=A;
			ic++;	
			A=0;
		}
	}
	sleep(rank);
	quicksort(C,0,ic-1,L);
	printf("---Table C---\n");
	print_table(C,ic,4,L); 
}

void check_inc_Q(){
	int s,z=0;
	int A=0;
	double a;
	L = (Nq / P);
	for(s=0;s<=L*3;s++){
		// if its not out of limits
		if(iq>=L){
			break;
		}
		// printf("%d %d %d\n",z*L+s,(z+1)*L+s,(z+2)*L+s);
		// Jump to the next subtable when zeros
		if((Qi[z*L+s]==0 && Qi[(z+1)*L+s]==0 && Qi[(z+2)*L+s]==0) || s==L*3){
			z+=3;
			if(z<(numtasks-1)*3){
				s=-1;
				continue;
			}else{
				break;
			}
			
		}
		// if its ok for that process
		if(Qi[z*L+s] < nbh && Qi[(z+1)*L+s] < mbh && Qi[(z+2)*L+s] < kbh && Qi[z*L+s] >= nbl && Qi[(z+1)*L+s] >= mbl && Qi[(z+2)*L+s] >= kbl){
			Q[0*L+iq]=0;
			Q[1*L+iq]=0;
			Q[2*L+iq]=0;
			Q[3*L+iq]=0;
	// Find x coordinate
			for(a = nbl ; a < nbh ; a = a + (nbh-nbl)/n){
				if(Qi[z*L+s]<a){
					break;
				}
				A++;
			}
			A*=100;
	// Find y coordinate
			for(a = mbl ; a < mbh ; a = a + (mbh-nbl)/m){
				if(Qi[(z+1)*L+s]<a){
					break;
				}
				A++;
			}
			A*=100;
	// Find z coordinate
			for(a = kbl ; a < kbh ; a = a + (kbh-kbl)/k){
				if(Qi[(z+2)*L+s]<a){
					break;
				}
				A++;
			}
			Q[0*L+iq]=Qi[z*L+s];
			Q[1*L+iq]=Qi[(z+1)*L+s];
			Q[2*L+iq]=Qi[(z+2)*L+s];
			Q[3*L+iq]=A;
			iq++;	
			A=0;
		}
	}
	sleep(rank);
	quicksort(Q,0,iq-1,L);
	printf("---Table Q---\n");
	print_table(Q,iq,4,L); 


}

void make_grid(){

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
	srand((time.tv_sec * 1000*rank) + (time.tv_usec / (1000*rank)));

// Malloc tables
	
	L=Nc/P;
	C =(double *) malloc(L*4*sizeof(double));
	Co = (double *) malloc(L*3*sizeof(double));
	L=Nq/P;
	Q =(double *) malloc(L*4*sizeof(double*));
	Qo =(double *) malloc(L*3*sizeof(double*));
	// printf("# of boxes: %f %f %f \n",(nbh-nbl)/n,(mbh-mbl)/m,(kbh-kbl)/k);

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
			Co[0*L+jc]=0;
			Co[1*L+jc]=0;
			Co[2*L+jc]=0;

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
			Qo[0*L+jq]=0;
			Qo[1*L+jq]=0;
			Qo[2*L+jq]=0;
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
