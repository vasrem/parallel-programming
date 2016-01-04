
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

struct timeval startwtime, endwtime;
double seq_time;

void make_grid();
void check_inc_C();
void check_inc_Q();
void print_table(double *,int,int,int);
void quicksort(double *,int,int,int);
void search_nn();


int main(int argc, char **argv){
	int temp,i,s;
	float buf[6];
	MPI_Status *stats;
	MPI_Group workers;
	MPI_Comm workers_comm;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	stats = (MPI_Status *) malloc(numtasks*sizeof(MPI_Status));
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
	MPI_Comm mc; // Communication group
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

	for(i=0;i<(Nc/P)*3*(numtasks-1);i++){
		Ci[i]=0;
	}
	for(i=0;i<(Nc/P)*3*(numtasks-1);i++){
		Qi[i]=0;
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
		printf("--------Info--------\n");
		printf("# of processes: %d\n",P);
		printf("size of C     : %d\n",Nc);
		printf("size of Q     : %d\n",Nq);
		printf("size of grid  : %d\n",gs);
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
		gettimeofday (&startwtime, NULL);
		make_grid();
		
		if(numtasks>2){
			MPI_Barrier(mc);
			MPI_Allgather(&Co[0],(Nc/P)*3,MPI_DOUBLE,&Ci[0],(Nc/P)*3,MPI_DOUBLE,mc);	
			MPI_Barrier(mc);
			MPI_Allgather(&Qo[0],(Nq/P)*3,MPI_DOUBLE,&Qi[0],(Nq/P)*3,MPI_DOUBLE,mc);
			// MPI_Barrier(mc);
			// for(i=1;i<numtasks;i++){
			// 	if(rank==i){
			// 		L=Nc/P;
			// 		printf("\n\t-------RANK=%d--------\n",rank);
			// 		printf("\n---Table C---\n");
			// 		print_table(C,ic,4,L); 
			// 		printf("---Table Co---\n");
			// 		print_table(Co,jc,3,L);
			// 		L=Nq/P;
			// 		printf("---Table Q---\n");
			// 		print_table(Q,iq,4,L); 
			// 		printf("---Table Qo---\n");
			// 		print_table(Qo,jq,3,L);
			// 	}
			// 	MPI_Barrier(mc);
			// }
			// MPI_Barrier(mc);
			// if(rank==1){
			// 	printf("\n");
			// 	for(i=0;i<(Nc/P)*3*(numtasks-1);i++){
			// 		printf("%f\t",Ci[i]);
			// 	}
			// 	printf("\n\n");
			// 	for(i=0;i<(Nq/P)*3*(numtasks-1);i++){
			// 		printf("%f\t",Qi[i]);
			// 	}
			// 	printf("\n");
			// }
			// MPI_Barrier(mc);
			// if(rank==5){
			// 	printf("\n");
			// 	for(i=0;i<(Nc/P)*3*(numtasks-1);i++){
			// 		printf("%f\t",Ci[i]);
			// 	}
			// 	printf("\n\n");
			// 	for(i=0;i<(Nq/P)*3*(numtasks-1);i++){
			// 		printf("%f\t",Qi[i]);
			// 	}
			// 	printf("\n");
			// }
			MPI_Barrier(mc);
			check_inc_C();
			MPI_Barrier(mc);
			check_inc_Q();
			MPI_Barrier(mc);
			// printf("#%d ic=%d\n",rank,ic);
			free(Ci);
			free(Qi);
			free(Co);
			free(Qo);
			// for(i=1;i<numtasks;i++){
			// 	if(rank==i){
			// 		printf("\n\t-------RANK=%d--------\n",rank);
			// 		printf("---Table C---\n");
			// 		print_table(C,ic,4,L); 
			// 		printf("---Table Q---\n");
			// 		print_table(Q,iq,4,L); 
			// 	}
			// 	MPI_Barrier(mc);
			// }
		}
		gettimeofday (&endwtime, NULL);
		if(rank==1){
			seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
				+ endwtime.tv_sec - startwtime.tv_sec);

			printf("Grid creation time : %f\n",seq_time);
		}

		// Search
		gettimeofday (&startwtime, NULL);
		search_nn();
		MPI_Barrier(mc);
		gettimeofday (&endwtime, NULL);
		if(rank==1){
			seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
				+ endwtime.tv_sec - startwtime.tv_sec);

			printf("Search nearest neighbor time : %f\n",seq_time);
		}
	}
	
	MPI_Finalize();

}
void search_nn(){
	// sleep(rank);
	// printf("Process #%d\n",rank);
	int i,j,s,z,c;
	double temp;
	int target;
	double *Cxl,*Cyl,*Czl;
	double *Cxh,*Cyh,*Czh;
	MPI_Request *reqs;
	reqs = (MPI_Request *) malloc(12*sizeof(MPI_Request));
	MPI_Status *stats;
	stats = (MPI_Status *) malloc(6*sizeof(MPI_Status));
	c=0;
	// Initialize the incoming tables
	Cxl =(double *) malloc((Nc/P)*4*sizeof(double));
	Cyl =(double *) malloc((Nc/P)*4*sizeof(double));
	Czl =(double *) malloc((Nc/P)*4*sizeof(double));
	Cxh =(double *) malloc((Nc/P)*4*sizeof(double));
	Cyh =(double *) malloc((Nc/P)*4*sizeof(double));
	Czh =(double *) malloc((Nc/P)*4*sizeof(double));

	// MPI Send Recieves

	// xh send
	target = rank + ( (ko/k) * (mo/m) );
	if(target<numtasks){
		// printf("Sending xh to rank #%d\n",target);
		MPI_Isend(&C[0],Nc/P,MPI_DOUBLE,target,1+(10*rank),MPI_COMM_WORLD,&reqs[0]);
	}

	// xl send
	target = rank - ( (ko/k) * (mo/m) );
	if(target>0){
		// printf("Sending xl to rank #%d\n",target);
		MPI_Isend(&C[0],Nc/P,MPI_DOUBLE,target,2+(10*rank),MPI_COMM_WORLD,&reqs[1]);
	}

	// yh send
	target = rank + (ko/k);
	if(target<numtasks){
		// printf("Sending yh to rank #%d\n",target);
		MPI_Isend(&C[0],Nc/P,MPI_DOUBLE,target,3+(10*rank),MPI_COMM_WORLD,&reqs[2]);
	}

	// yl send
	target = rank - (ko/k);
	if(target>0){
		// printf("Sending yl to rank #%d\n",target);
		MPI_Isend(&C[0],Nc/P,MPI_DOUBLE,target,4+(10*rank),MPI_COMM_WORLD,&reqs[3]);
	}

	// zh send
	target = rank + 1;
	if(target<numtasks){
		// printf("Sending zh to rank #%d\n",target);
		MPI_Isend(&C[0],Nc/P,MPI_DOUBLE,target,5+(10*rank),MPI_COMM_WORLD,&reqs[4]);
	}

	// zl send
	target = rank - 1;
	if(target>0){
		// printf("Sending zl to rank #%d\n",target);
		MPI_Isend(&C[0],Nc/P,MPI_DOUBLE,target,6+(10*rank),MPI_COMM_WORLD,&reqs[5]);
	}


	// if there is xh
	target = rank + ( (ko/k) * (mo/m) );
	if(target<numtasks){
		MPI_Irecv(&Cxh[0],Nc/P,MPI_DOUBLE,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[6]);
	}

	// if there is xl
	target = rank - ( (ko/k) * (mo/m) );
	if(target>0){
		MPI_Irecv(&Cxl[0],Nc/P,MPI_DOUBLE,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[7]);
	}

	// if there is yh
	target = rank + (ko/k);
	if(target<numtasks){
		MPI_Irecv(&Cyh[0],Nc/P,MPI_DOUBLE,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[8]);
	}

	// if there is yl
	target = rank - (ko/k);
	if(target>0){
		MPI_Irecv(&Cyl[0],Nc/P,MPI_DOUBLE,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[9]);
	}

	// if there is zh
	target = rank + 1;
	if(target<numtasks){
		MPI_Irecv(&Czh[0],Nc/P,MPI_DOUBLE,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[10]);
	}

	// if there is zl
	target = rank - 1;
	if(target>0){
		MPI_Irecv(&Czl[0],Nc/P,MPI_DOUBLE,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[11]);
	}


	// min value and index.
	double min=1;
	int mini=0;
	double min_n[3];
	for(i=0;i<iq;i++){
		// printf("Check for -> %f %f %f %f\n",Q[0*(Nq/P)+i],Q[1*(Nq/P)+i],Q[2*(Nq/P)+i],Q[3*(Nq/P)+i]);
		// avoid to check to some boxes.
		if(i>0){
			if(Q[3*(Nq/P)+i]==Q[3*(Nq/P)+i-1]){
				// printf("mphka\n");
				c-=z;
			}
			z=-1;
		}
		for(j=0;j<ic;j++){
			z++;
			// Check if its in the same box.
			if(Q[3*(Nq/P)+i]==C[3*(Nc/P)+j]){
				temp=sqrt(pow(C[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(C[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(C[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
				// printf("%f\n",temp);
				if(min>temp){
					min=temp;
					min_n[0]=C[0*(Nc/P)+j];
					min_n[1]=C[1*(Nc/P)+j];
					min_n[2]=C[2*(Nc/P)+j];
					mini=j;
				}
			}
			else{
				c=j;
				break;
			}
			// else{
			// 	break;
			// }
			// the min distance with the distance of Q[{0,1,2}*5+i] from the bounds.
		}
		if(j==ic){
			z--;
		}
		// printf("Bound nbh:%f %f is %d\n",fabs( Q[0*(Nc/P)+i] - C[0*(Nq/P)+mini] ),fabs( Q[0*(Nq/P)+i] - nbh ),fabs( Q[0*(Nc/P)+i] - C[0*(Nq/P)+mini] )>fabs( Q[0*(Nq/P)+i] - nbh ) );
		// printf("Bound nbl:%f %f is %d\n",fabs( Q[0*(Nc/P)+i] - C[0*(Nq/P)+mini] ),fabs( Q[0*(Nq/P)+i] - nbl ),fabs( Q[0*(Nc/P)+i] - C[0*(Nq/P)+mini] )>fabs( Q[0*(Nq/P)+i] - nbl ));
		// printf("Bound mbh:%f %f is %d\n",fabs( Q[1*(Nc/P)+i] - C[1*(Nq/P)+mini] ),fabs( Q[1*(Nq/P)+i] - mbh ),fabs( Q[1*(Nc/P)+i] - C[1*(Nq/P)+mini] )>fabs( Q[1*(Nq/P)+i] - mbh ) );
		// printf("Bound mbl:%f %f is %d\n",fabs( Q[1*(Nc/P)+i] - C[1*(Nq/P)+mini] ),fabs( Q[1*(Nq/P)+i] - mbl ),fabs( Q[1*(Nc/P)+i] - C[1*(Nq/P)+mini] )>fabs( Q[1*(Nq/P)+i] - mbl ) );
		// printf("Bound kbh:%f %f is %d\n",fabs( Q[2*(Nc/P)+i] - C[2*(Nq/P)+mini] ),fabs( Q[2*(Nq/P)+i] - kbh ),fabs( Q[2*(Nc/P)+i] - C[2*(Nq/P)+mini] )>fabs( Q[2*(Nq/P)+i] - kbh ) );
		// printf("Bound kbl:%f %f is %d\n",fabs( Q[2*(Nc/P)+i] - C[2*(Nq/P)+mini] ),fabs( Q[2*(Nq/P)+i] - kbl ),fabs( Q[2*(Nc/P)+i] - C[2*(Nq/P)+mini] )>fabs( Q[2*(Nq/P)+i] - kbl ) );
		if( fabs( Q[0*(Nc/P)+i] - C[0*(Nq/P)+mini] ) > fabs( Q[0*(Nq/P)+i] - nbh ) ){
			// na koitaw an uparxei auto to kouti se auto to process
			// ---- an uparxei tote vriskw to kainourgio min sugkrinontas to Q[0*5+i] me ola ta C tou allou koutiou 
			// kai sigkrinw me to min p vrika parapanw[ line :104 ]
			// ---- an den uparxei tote 8a prepei na zitaw apo to process p einai dipla m, analoga me ta bounds, to C tou kai na sigkrinw to Q[0*5+i]
			// me to C tou swstou koutiou tou allou process.
			// printf("check xxh %i =%d\n",i,(int)(Q[3*(Nq/P)+i]+10000)/10000);
			if((int)(Q[3*(Nq/P)+i]+10000)/10000<=(no/n)){
				// printf("Maybe nearest to %f\n",Q[3*(Nq/P)+i]+10000);
				// check sto kouti me Q[3*(Nq/P)+i]+10000 sto C[3*5+j]
				for(j=0;j<ic;j++){
					if((Q[3*(Nq/P)+i]+10000)==C[3*(Nc/P)+j]){
						temp=sqrt(pow(C[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(C[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(C[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
						// printf("%f\n",temp);
						if(min>temp){
							min=temp;
							min_n[0]=C[0*(Nc/P)+j];
							min_n[1]=C[1*(Nc/P)+j];
							min_n[2]=C[2*(Nc/P)+j];
							mini=j;
						}
					}
				}
			}else{
				target = rank + ( (ko/k) * (mo/m) );
				if(target<numtasks){
					// mpi rcv to rank+(mxk) opou m kai k se posa koutia spaei i ka8e diastasi ka8e fora.
					// psakse sta akriana apo ta mikra
					MPI_Wait(&reqs[6],&stats[0]);
					// Go to the box with the same y z and x = 1
					// target =( ( (int) (Q[3*(Nq/P)+i]) ) % 10000 ) + 10000;
					for(j=0;j<(ic/(n*m*k));j++){
						// if(target == Cxh[3*(Nc/P)+j]){
							temp=sqrt(pow(Cxh[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(Cxh[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(Cxh[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
							if(min>temp){
								min=temp;
								min_n[0]=Cxh[0*(Nc/P)+j];
								min_n[1]=Cxh[1*(Nc/P)+j];
								min_n[2]=Cxh[2*(Nc/P)+j];
								mini=j;
							}
						// }
					}
				}
			}
		}
		if( fabs( Q[0*(Nc/P)+i] - C[0*(Nq/P)+mini] ) > fabs( Q[0*(Nq/P)+i] - nbl ) ){
			// printf("check xxl %i =%d\n",i,(int)(Q[3*(Nq/P)+i]-10000)/10000);
				// printf("Maybe nearest to %f\n",Q[3*(Nq/P)+i]-10000);
			if((int)(Q[3*(Nq/P)+i]-10000)/10000>=1){
				// check sto kouti me Q[3*(Nq/P)+i]-10000 sto C[3*(Nc/P)+j] 
				for(j=0;j<ic;j++){
					if((Q[3*(Nq/P)+i]-10000)==C[3*(Nc/P)+j]){
						temp=sqrt(pow(C[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(C[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(C[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
						// printf("%f\n",temp);
						if(min>temp){
							min=temp;
							min_n[0]=C[0*(Nc/P)+j];
							min_n[1]=C[1*(Nc/P)+j];
							min_n[2]=C[2*(Nc/P)+j];
							mini=j;
						}
					}
				}
			}else{
				target = rank - ( (ko/k) * (mo/m) );
				if(target>0){
					// mpi rcv to rank-(mxk)
					// psakse sta akriana apo ta megala

					MPI_Wait(&reqs[7],&stats[1]);
					for(j=(ic/(n*m*k))-1;j>=ic-(ic/(n*m*k));j--){
						// if(target == Cxh[3*(Nc/P)+j]){
							temp=sqrt(pow(Cxl[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(Cxl[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(Cxl[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
							if(min>temp){
								min=temp;
								min_n[0]=Cxl[0*(Nc/P)+j];
								min_n[1]=Cxl[1*(Nc/P)+j];
								min_n[2]=Cxl[2*(Nc/P)+j];
								mini=j;
							}
						// }
					}
				}
			}
		}
		//rdy
		if( fabs( Q[1*(Nc/P)+i] - C[1*(Nq/P)+mini] ) > fabs( Q[1*(Nq/P)+i] - mbh ) ){
			// printf("Maybe nearest to %f\n",Q[3*(Nq/P)+i]+100);
			// printf("check yyh %i =%d\n",i,((((int)Q[3*(Nq/P)+i])%10000)+100)/100);
			if(((((int)Q[3*(Nq/P)+i])%10000)+100)/100<=(mo/m)){
				// check sto kouti me Q[3*(Nq/P)+i]+100 sto C[3*(Nc/P)+j]
				for(j=0;j<ic;j++){
					if((Q[3*(Nq/P)+i]+100)==C[3*(Nc/P)+j]){
						temp=sqrt(pow(C[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(C[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(C[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
						// printf("%f\n",temp);
						if(min>temp){
							min=temp;
							min_n[0]=C[0*(Nc/P)+j];
							min_n[1]=C[1*(Nc/P)+j];
							min_n[2]=C[2*(Nc/P)+j];
							mini=j;
						}
					}
				}
			}else{
				target = rank + (ko/k);
				if(target<numtasks){
					// mpi rcv to rank+k
					// psakse sta akriana apo ta mikra
					MPI_Wait(&reqs[8],&stats[2]);
					for(j=0;j<(ic/(n*m*k));j++){
						// if(target == Cxh[3*(Nc/P)+j]){
							temp=sqrt(pow(Cyh[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(Cyh[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(Cyh[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
							if(min>temp){
								min=temp;
								min_n[0]=Cyh[0*(Nc/P)+j];
								min_n[1]=Cyh[1*(Nc/P)+j];
								min_n[2]=Cyh[2*(Nc/P)+j];
								mini=j;
							}
						// }
					}
				}
			}
		}
		if( fabs( Q[1*(Nc/P)+i] - C[1*(Nq/P)+mini] ) > fabs( Q[1*(Nq/P)+i] - mbl ) ){
			// printf("Maybe nearest to %f\n",Q[3*(Nq/P)+i]-100);
			// printf("check yyl %i =%d\n",i,((((int)Q[3*(Nq/P)+i])%10000)-100)/100);
			if(((((int)Q[3*(Nq/P)+i])%10000)-100)/100>=1){
				// check sto kouti me Q[3*(Nq/P)+i]-100 sto C[3*(Nc/P)+j]
				for(j=0;j<ic;j++){
					if((Q[3*(Nq/P)+i]-100)==C[3*(Nc/P)+j]){
						temp=sqrt(pow(C[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(C[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(C[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
						// printf("%f\n",temp);
						if(min>temp){
							min=temp;
							min_n[0]=C[0*(Nc/P)+j];
							min_n[1]=C[1*(Nc/P)+j];
							min_n[2]=C[2*(Nc/P)+j];
							mini=j;
						}
					}
				}
			}else{
				target = rank - (ko/k);
				if(target>0){
					// mpi rcv to rank-k
					// psakse sta akriana apo ta megala
					MPI_Wait(&reqs[9],&stats[3]);
					for(j=(ic/(n*m*k))-1;j>=ic-(ic/(n*m*k));j--){
						// if(target == Cxh[3*(Nc/P)+j]){
							temp=sqrt(pow(Cyl[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(Cyl[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(Cyl[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
							if(min>temp){
								min=temp;
								min_n[0]=Cyl[0*(Nc/P)+j];
								min_n[1]=Cyl[1*(Nc/P)+j];
								min_n[2]=Cyl[2*(Nc/P)+j];
								mini=j;
							}
						// }
					}
				}
			}
		}
		if( fabs( Q[2*(Nc/P)+i] - C[2*(Nq/P)+mini] ) > fabs( Q[2*(Nq/P)+i] - kbh ) ){
			// printf("Maybe nearest to %f\n",Q[3*5+i]+1);
			// printf("check zzh %i =%d\n",i,((((int)Q[3*(Nq/P)+i])%100)+1));
			if(((((int)Q[3*(Nq/P)+i])%100)+1)<=(k/ko)){
				// check sto kouti me Q[3*5+i]+1 sto C[3*(Nc/P)+j]
				for(j=0;j<ic;j++){
					if((Q[3*(Nq/P)+i]+1)==C[3*(Nc/P)+j]){
						temp=sqrt(pow(C[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(C[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(C[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
						// printf("%f\n",temp);
						if(min>temp){
							min=temp;
							min_n[0]=C[0*(Nc/P)+j];
							min_n[1]=C[1*(Nc/P)+j];
							min_n[2]=C[2*(Nc/P)+j];
							mini=j;
						}
					}
				}
			}else{
				target = rank + 1;
				if(target<numtasks){
					// mpi rcv to rank+1
					// psakse sto idio kouti me to C[3*5+i]
					MPI_Wait(&reqs[10],&stats[4]);
					for(j=0;j<(ic/(n*m*k));j++){
						// if(target == Cxh[3*(Nc/P)+j]){
							temp=sqrt(pow(Czh[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(Czh[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(Czh[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
							if(min>temp){
								min=temp;
								min_n[0]=Czh[0*(Nc/P)+j];
								min_n[1]=Czh[1*(Nc/P)+j];
								min_n[2]=Czh[2*(Nc/P)+j];
								mini=j;
							}
						// }
					}
				}
			}
		}
		if( fabs( Q[2*(Nc/P)+i] - C[2*(Nq/P)+mini] ) > fabs( Q[2*(Nq/P)+i] - kbl ) ){
			// printf("Maybe nearest to %f\n",Q[3*5+i]-1);
			// printf("check zzl %i =%d\n",i,((((int)Q[3*(Nq/P)+i])%100)-1));
			if(((((int)Q[3*(Nq/P)+i])%100)-1)>=1){
				// check sto kouti me Q[3*5+i]-1 sto C[3*(Nc/P)+j]
				for(j=0;j<ic;j++){
					if((Q[3*(Nq/P)+i]-1)==C[3*(Nc/P)+j]){
						temp=sqrt(pow(C[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(C[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(C[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
						// printf("%f\n",temp);
						if(min>temp){
							min=temp;
							min_n[0]=C[0*(Nc/P)+j];
							min_n[1]=C[1*(Nc/P)+j];
							min_n[2]=C[2*(Nc/P)+j];
							mini=j;
						}
					}
				}
			}else{
				target = rank - 1;
				if(target>0){
					// mpi rcv to rank-1
					// psakse sto idio kouti me to C[3*5+i]
					MPI_Wait(&reqs[11],&stats[5]);
					for(j=(ic/(n*m*k))-1;j>=ic-(ic/(n*m*k));j--){
						// if(target == Cxh[3*(Nc/P)+j]){
							temp=sqrt(pow(Czl[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(Czl[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(Czl[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
							if(min>temp){
								min=temp;
								min_n[0]=Czl[0*(Nc/P)+j];
								min_n[1]=Czl[1*(Nc/P)+j];
								min_n[2]=Czl[2*(Nc/P)+j];
								mini=j;
							}
						// }
					}
				}
			}
		}
		min=1;
		mini=0;
		// printf("min=%f %f %f",min_n[0],min_n[1],min_n[2]);
		// if(z==5){
		// 	// printf("Kontinoteros %d -> %f %f %f %f\n",mini,C[0*(Nq/P)+mini],C[1*(Nq/P)+mini],C[2*(Nq/P)+mini],C[3*(Nq/P)+mini]);
		// 	min=1;
		// 	mini=0;
		// 	z=0;
		// }
	}
}


void check_inc_C(){
	int s,z=0;
	int A=0;
	double a,vima;
	L = (Nc / P);
	// sleep(2*rank);
	double *d;
	d =(double *) malloc(3*sizeof(double));
	struct timeval time; 
	gettimeofday(&time,NULL);
	srand((time.tv_sec * 1000)*(rank+1) + (time.tv_usec / 1000)*(rank+1));
	for(s=0;s<=L;s++){
		// if its not out of limits
		if(ic>=L){
			break;
		}
		// printf("%d %d %d\n",z*L+s,(z+1)*L+s,(z+2)*L+s);
		// Jump to the next subtable when zeros
		// printf("rank %d checking ic=%d s=%d\n",rank,ic,s);
		// printf("Ci[%d][%d]=%f [%d]=%f [%d]=%f\n",s,z,Ci[z*L+s],z+1,Ci[(z+1)*L+s],z+2,Ci[(z+2)*L+s]);
		if((Ci[z*L+s]==0 && Ci[(z+1)*L+s]==0 && Ci[(z+2)*L+s]==0) || s==L){
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
			vima = nbh-nbl;
			for(a = nbl ; a < nbh ; a = a + (nbh-nbl)/n){
				if(Ci[z*L+s]<a){
					break;
				}
				A++;
			}
			A*=100;
	// Find y coordinate
			for(a = mbl ; a < mbh ; a = a + (mbh-mbl)/m){
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
	for(s=ic;s<L;s++){
		d[0] =nbl +((double) rand()/RAND_MAX)*(nbh-nbl);
		d[1] =mbl +((double) rand()/RAND_MAX)*(mbh-mbl);
		d[2] =kbl +((double) rand()/RAND_MAX)*(kbh-kbl);
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
	}

	// sleep(2*rank);
	quicksort(C,0,ic-1,L);
	
}

void check_inc_Q(){
	int s,z=0;
	int A=0;
	double a;
	L = (Nq / P);

	double *d;
	d =(double *) malloc(3*sizeof(double));

	struct timeval time; 
	gettimeofday(&time,NULL);
	srand((time.tv_sec * 1000)*(rank+1) + (time.tv_usec / 1000)*(rank+1));

	for(s=0;s<=L;s++){
		// if its not out of limits
		if(iq>=L){
			break;
		}
		// printf("%d %d %d\n",z*L+s,(z+1)*L+s,(z+2)*L+s);
		// Jump to the next subtable when zeros
		if((Qi[z*L+s]==0 && Qi[(z+1)*L+s]==0 && Qi[(z+2)*L+s]==0) || s==L){
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
			for(a = mbl ; a < mbh ; a = a + (mbh-mbl)/m){
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
	for(s=iq;s<L;s++){
		d[0] =nbl +((double) rand()/RAND_MAX)*(nbh-nbl);
		d[1] =mbl +((double) rand()/RAND_MAX)*(mbh-mbl);
		d[2] =kbl +((double) rand()/RAND_MAX)*(kbh-kbl);
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
	}
	// sleep(2*rank);
	quicksort(Q,0,iq-1,L);
	


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
	srand((time.tv_sec * 1000)*(rank+1) + (time.tv_usec / 1000)*(rank+1));

// Malloc tables
	
	L=Nc/P;
	C =(double *) malloc(L*4*sizeof(double));
	Co = (double *) malloc(L*3*sizeof(double));
	L=Nq/P;
	Q =(double *) malloc(L*4*sizeof(double));
	Qo =(double *) malloc(L*3*sizeof(double));
	// printf("# of boxes: %f %f %f \n",(nbh-nbl)/n,(mbh-mbl)/m,(kbh-kbl)/k);
	//printf("Lc=%d  Lq=%d\n",Nc/P,Nq/P);
	// zero to everything
	for(s=0; s < L*4;s++){
		C[s]=0;
		Q[s]=0;
		if(s<L*3){
			Co[s]=0;
			Qo[s]=0;
		}
	}
	// Generate C
	L=Nc/P;
	for( s = 0 ; s < L ; s++ ){
		d[0] = (double) rand()/RAND_MAX;
		d[1] = (double) rand()/RAND_MAX;
		d[2] = (double) rand()/RAND_MAX;
		// printf("C[%d]:%0.10f\t%0.10f\t%0.10f\t\n",rank,d[0],d[1],d[2]);
	// if the number is ok for this process
		if( d[0] < nbh && d[1] < mbh && d[2] < kbh && d[0] >= nbl && d[1] >= mbl && d[2] >= kbl ){
			// C[0*L+ic]=0;
			// C[1*L+ic]=0;
			// C[2*L+ic]=0;
			// C[3*L+ic]=0;
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
			// Co[0*L+jc]=0;
			// Co[1*L+jc]=0;
			// Co[2*L+jc]=0;
		}
	}

// Generate Q

	L=Nq/P;
	for( s = 0 ; s < L ; s++ ){


		d[0] = (double) rand()/RAND_MAX;
		d[1] = (double) rand()/RAND_MAX;
		d[2] = (double) rand()/RAND_MAX;
	 	// printf("Q[%d]:%0.10f\t%0.10f\t%0.10f\t\n",rank,d[0],d[1],d[2]);

	// if the number is ok for this process
		if( d[0] < nbh && d[1] < mbh && d[2] < kbh && d[0] >= nbl && d[1] >= mbl && d[2] >= kbl ){
			// Q[0*L+iq]=0;
			// Q[1*L+iq]=0;
			// Q[2*L+iq]=0;
			// Q[3*L+iq]=0;
			
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
			// Qo[0*L+jq]=0;
			// Qo[1*L+jq]=0;
			// Qo[2*L+jq]=0;
		}

	}


	quicksort(C,0,ic-1,Nc/P);
	quicksort(Q,0,iq-1,Nq/P);
	// sleep(rank);

	// printf("rank=%d\n",rank);
	// for(s=0;s<L*4;s++){
	// 	printf("%f ",C[s]);
	// }
	// printf("---CO---\n");
	// for(s=0;s<L*3;s++){
	// 	printf("%f ",Co[s]);
	// }
	

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
