
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

int *indexOfBoxC; // Starting index of specific Box in C


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
double grid_time,search_time;

void make_grid();
void check_inc_C();
void check_inc_Q();
void print_table(double *,int,int,int);
void quicksort(double *,int,int,int);
void search_nn();
void search_serial();


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

	
	if(rank==0){
		// printf("Number of active processes is %d\n",numtasks);
		// printf("%d %d %d %d\n",Nc,Nq,gs,P);
		// printf("n=%f m=%f k=%f\n",n,m,k);
		// printf("no=%f mo=%f ko=%f\n",no,mo,ko);
		printf("--------Info--------\n");
		printf("# of processes: %d\t-> %d\n",atoi(argv[4]),P);
		printf("size of C     : %d\t-> %d\n",atoi(argv[1]),Nc);
		printf("size of Q     : %d\t-> %d\n",atoi(argv[2]),Nq);
		printf("size of grid  : %d\t-> %d\n",atoi(argv[3]),gs);
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
		gettimeofday (&startwtime, NULL);
		make_grid();
		
		if(numtasks>2){
			MPI_Barrier(mc);
			MPI_Allgather(&Co[0],(Nc/P)*3,MPI_DOUBLE,&Ci[0],(Nc/P)*3,MPI_DOUBLE,mc);	
			MPI_Barrier(mc);
			MPI_Allgather(&Qo[0],(Nq/P)*3,MPI_DOUBLE,&Qi[0],(Nq/P)*3,MPI_DOUBLE,mc);
			MPI_Barrier(mc);

			check_inc_C();
			check_inc_Q();
			MPI_Barrier(mc);
			MPI_Barrier(mc);
			free(Ci);
			free(Qi);
			free(Co);
			free(Qo);
		}
		gettimeofday (&endwtime, NULL);
		if(rank==1){
			grid_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
				+ endwtime.tv_sec - startwtime.tv_sec);

			printf("Grid creation time : %f\n",grid_time);
		}

		// Search

		gettimeofday (&startwtime, NULL);
		if(numtasks==2){
			search_serial();
		}else{
			search_nn();
		}
		
		MPI_Barrier(mc);
		gettimeofday (&endwtime, NULL);
		if(rank==1){
			search_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
				+ endwtime.tv_sec - startwtime.tv_sec);

			printf("Search nearest neighbor time : %f\n",search_time);

			FILE *f = fopen("times.txt", "a");
			if (f == NULL)
			{
				printf("Error opening file!\n");
				exit(1);
			}
			fprintf(f,"%d\t-\t%d\t-\t%d\t->\t%f\t%f\n",atoi(argv[1]),atoi(argv[3]),P,grid_time,search_time);
			fclose(f);
		}
	}
	
	MPI_Finalize();

}

void search_serial(){
	double min=1;
	int mini=-1;
	double min_n[3];
	double temp;
	int i,j;
	for(i=0;i<iq;i++){
		for(j=0;j<ic;j++){
			temp=sqrt(pow(C[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(C[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(C[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
			if(min>temp){
				min=temp;
				min_n[0]=C[0*(Nc/P)+j];
				min_n[1]=C[1*(Nc/P)+j];
				min_n[2]=C[2*(Nc/P)+j];
				mini=j;
			}
		}
	}
}

void search_nn(){

	int i,j,s,z,c;
	int e1,e2,e3;
	double temp;
	int target;
	int target2;
	int xx;	//x box
	int yy;	//y box
	int zz;	//z box
	int v=2; // TEST RANK
	double *Cxl,*Cyl,*Czl;
	double *Cxh,*Cyh,*Czh;
	

	int *indexOfBoxCxl,*indexOfBoxCyl,*indexOfBoxCzl;
	int *indexOfBoxCxh,*indexOfBoxCyh,*indexOfBoxCzh;

	int done[6];
	for(i=0;i<6;i++){
		done[i]=0;
	}

	MPI_Request *reqs;
	reqs = (MPI_Request *) malloc(24*sizeof(MPI_Request));
	MPI_Status *stats;
	stats = (MPI_Status *) malloc(12*sizeof(MPI_Status));

	// Initialize the incoming tables
	Cxl =(double *) malloc((Nc/P)*4*sizeof(double));
	Cyl =(double *) malloc((Nc/P)*4*sizeof(double));
	Czl =(double *) malloc((Nc/P)*4*sizeof(double));
	Cxh =(double *) malloc((Nc/P)*4*sizeof(double));
	Cyh =(double *) malloc((Nc/P)*4*sizeof(double));
	Czh =(double *) malloc((Nc/P)*4*sizeof(double));

	indexOfBoxCxl=(int *) malloc(999999*sizeof(int));
	indexOfBoxCyl=(int *) malloc(999999*sizeof(int));
	indexOfBoxCzl=(int *) malloc(999999*sizeof(int));
	indexOfBoxCxh=(int *) malloc(999999*sizeof(int));
	indexOfBoxCyh=(int *) malloc(999999*sizeof(int));
	indexOfBoxCzh=(int *) malloc(999999*sizeof(int));


	// xh send
	target = rank + ( (ko/k) * (mo/m) );
	if(target<numtasks){
		// printf("Sending xh to rank #%d\n",target);
		MPI_Isend(&C[0],Nc/P,MPI_DOUBLE,target,1+(10*rank),MPI_COMM_WORLD,&reqs[0]);
		MPI_Isend(&indexOfBoxC[0],999999,MPI_INT,target,1+(1000*rank),MPI_COMM_WORLD,&reqs[1]);
	}

	// xl send
	target = rank - ( (ko/k) * (mo/m) );
	if(target>0){
		// printf("Sending xl to rank #%d\n",target);
		MPI_Isend(&C[0],Nc/P,MPI_DOUBLE,target,2+(10*rank),MPI_COMM_WORLD,&reqs[2]);
		MPI_Isend(&indexOfBoxC[0],999999,MPI_INT,target,2+(1000*rank),MPI_COMM_WORLD,&reqs[3]);
	}

	// yh send
	target = rank + (ko/k);
	if(target<numtasks){
		// printf("Sending yh to rank #%d\n",target);
		MPI_Isend(&C[0],Nc/P,MPI_DOUBLE,target,3+(10*rank),MPI_COMM_WORLD,&reqs[4]);
		MPI_Isend(&indexOfBoxC[0],999999,MPI_INT,target,3+(1000*rank),MPI_COMM_WORLD,&reqs[5]);
	}

	// yl send
	target = rank - (ko/k);
	if(target>0){
		// printf("Sending yl to rank #%d\n",target);
		MPI_Isend(&C[0],Nc/P,MPI_DOUBLE,target,4+(10*rank),MPI_COMM_WORLD,&reqs[6]);
		MPI_Isend(&indexOfBoxC[0],999999,MPI_INT,target,4+(1000*rank),MPI_COMM_WORLD,&reqs[7]);
	}

	// zh send
	target = rank + 1;
	if(target<numtasks){
		// printf("Sending zh to rank #%d\n",target);
		MPI_Isend(&C[0],Nc/P,MPI_DOUBLE,target,5+(10*rank),MPI_COMM_WORLD,&reqs[8]);
		MPI_Isend(&indexOfBoxC[0],999999,MPI_INT,target,5+(1000*rank),MPI_COMM_WORLD,&reqs[9]);
	}

	// zl send
	target = rank - 1;
	if(target>0){
		// printf("Sending zl to rank #%d\n",target);
		MPI_Isend(&C[0],Nc/P,MPI_DOUBLE,target,6+(10*rank),MPI_COMM_WORLD,&reqs[10]);
		MPI_Isend(&indexOfBoxC[0],999999,MPI_INT,target,6+(1000*rank),MPI_COMM_WORLD,&reqs[11]);
	}

	// if there is xh
	target = rank + ( (ko/k) * (mo/m) );
	if(target<numtasks){
		MPI_Irecv(&Cxh[0],Nc/P,MPI_DOUBLE,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[12]);
		MPI_Irecv(&indexOfBoxCxh[0],999999,MPI_INT,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[13]);
	}

	// if there is xl
	target = rank - ( (ko/k) * (mo/m) );
	if(target>0){
		MPI_Irecv(&Cxl[0],Nc/P,MPI_DOUBLE,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[14]);
		MPI_Irecv(&indexOfBoxCzl[0],999999,MPI_INT,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[15]);
	}

	// if there is yh
	target = rank + (ko/k);
	if(target<numtasks){
		MPI_Irecv(&Cyh[0],Nc/P,MPI_DOUBLE,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[16]);
		MPI_Irecv(&indexOfBoxCyh[0],999999,MPI_INT,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[17]);
	}

	// if there is yl
	target = rank - (ko/k);
	if(target>0){
		MPI_Irecv(&Cyl[0],Nc/P,MPI_DOUBLE,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[18]);
		MPI_Irecv(&indexOfBoxCyl[0],999999,MPI_INT,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[19]);
	}

	// if there is zh
	target = rank + 1;
	if(target<numtasks){
		MPI_Irecv(&Czh[0],Nc/P,MPI_DOUBLE,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[20]);
		MPI_Irecv(&indexOfBoxCzh[0],999999,MPI_INT,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[21]);
	}

	// if there is zl
	target = rank - 1;
	if(target>0){
		MPI_Irecv(&Czl[0],Nc/P,MPI_DOUBLE,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[22]);
		MPI_Irecv(&indexOfBoxCzl[0],999999,MPI_INT,target,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[23]);
	}

	// min value and index.
	double min=1;
	int mini=-1;
	double min_n[3];
	
	for(i=0;i<iq;i++){
		temp=0;
		// Checking my box
		for(j=indexOfBoxC[(int)Q[3*(Nq/P)+i]];j<ic;j++){
			if(Q[3*(Nq/P)+i]==C[3*(Nc/P)+j]){
				temp=sqrt(pow(C[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(C[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(C[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
				if(min>temp){
					min=temp;
					min_n[0]=C[0*(Nc/P)+j];
					min_n[1]=C[1*(Nc/P)+j];
					min_n[2]=C[2*(Nc/P)+j];
					mini=j;
				}
			}else{
				break;
			}
		}

		xx=(int)Q[3*(Nq/P)+i]/10000;
		yy=(int)Q[3*(Nq/P)+i]%10000/100;
		zz=(int)Q[3*(Nq/P)+i]%100;	
		// Checking other boxes
		for(e1=xx-1;e1<=xx+1;e1++){
			for(e2=yy-1;e2<=yy+1;e2++){
				for(e3=zz-1;e3<=zz+1;e3++){

					if(e1==xx && e2==yy && e3==zz){
						continue;
					}

					if(e1<=n && e1>=1){

						if(e2<=m && e2>=1){

							if(e3<=k && e3>=1){

								target=10000*e1+100*e2+e3;
								if(e1==xx){
									if(e2==yy){
										if(e3==zz-1){
											done[0]=1;	// We will not search xx yy zz-1 to other process
										}else if(e3==zz+1){
											done[1]=1;	// We will not search xx yy zz+1 to other process
										}
									}else if(e2==yy-1){
										if(e3==zz){
											done[2]=1;	//We will not search xx yy-1 zz to other process
										}
									}else if(e2==yy+1){
										if(e3==zz){
											done[3]=1;	//We will not search xx yy+1 zz to other process
										}
									}
								}else if(e1==xx-1){
									if(e2==yy){
										if(e3==zz){
											done[4]=1;	//We will not search xx-1 yy zz to other process
										}
									}
								}else if(e1==xx+1){
									if(e2==yy){
										if(e3==zz){
											done[5]=1;	//We will not search xx+1 yy zz to other process
										}
									}
								}
								for(j=indexOfBoxC[target];j<ic;j++){
									if(target==C[3*(Nc/P)+j]){
										temp=sqrt(pow(C[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(C[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(C[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
										if(min>temp){
											min=temp;
											min_n[0]=C[0*(Nc/P)+j];
											min_n[1]=C[1*(Nc/P)+j];
											min_n[2]=C[2*(Nc/P)+j];
										}
									}else{
										break;
									}
								}
							} 

						}

					}

				}
			}
		}

		// Zl Check
		if(done[0]){
			target = rank - 1;
			if(target>0 && kbl>0){
				MPI_Wait(&reqs[22],&stats[0]);
				MPI_Wait(&reqs[23],&stats[1]);
				target=10000*xx+100*yy+k;
				for(j=indexOfBoxCzl[target];j<ic;j++){
					if(target==Czl[3*(Nc/P)+j]){
						temp=sqrt(pow(Czl[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(Czl[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(Czl[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
						if(min>temp){
							min=temp;
							min_n[0]=Czl[0*(Nc/P)+j];
							min_n[1]=Czl[1*(Nc/P)+j];
							min_n[2]=Czl[2*(Nc/P)+j];
						}
					}else{
						break;
					}	
				}

			}
		}
		// Zh Check
		if(done[1]){
			target = rank + 1;
			if(target<numtasks && kbh<1){
				MPI_Wait(&reqs[20],&stats[2]);
				MPI_Wait(&reqs[21],&stats[3]);
				target=10000*xx+100*yy+1;
				for(j=indexOfBoxCzh[target];j<ic;j++){
					if(target==Czh[3*(Nc/P)+j]){
						temp=sqrt(pow(Czh[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(Czh[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(Czh[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
						if(min>temp){
							min=temp;
							min_n[0]=Czh[0*(Nc/P)+j];
							min_n[1]=Czh[1*(Nc/P)+j];
							min_n[2]=Czh[2*(Nc/P)+j];
						}
					}else{
						break;
					}
				}

			}

		}
		// Yl Check
		if(done[2]){
			target = rank - (ko/k);
			if(target>0 && mbl>0){
				MPI_Wait(&reqs[18],&stats[4]);
				MPI_Wait(&reqs[19],&stats[5]);
				target=10000*xx+100*m+zz;
				for(j=indexOfBoxCyl[target];j<ic;j++){
					if(target==Cyl[3*(Nc/P)+j]){
						temp=sqrt(pow(Cyl[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(Cyl[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(Cyl[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
						if(min>temp){
							min=temp;
							min_n[0]=Cyl[0*(Nc/P)+j];
							min_n[1]=Cyl[1*(Nc/P)+j];
							min_n[2]=Cyl[2*(Nc/P)+j];
						}
					}else{
						break;
					}
				}

			}
		}
		// Yh Check
		if(done[3]){
			target = rank + (ko/k);
			if(target<numtasks && mbh<1){
				MPI_Wait(&reqs[16],&stats[6]);
				MPI_Wait(&reqs[17],&stats[7]);
				target=10000*xx+100*1+zz;
				for(j=indexOfBoxCyh[target];j<ic;j++){
					if(target==Cyh[3*(Nc/P)+j]){
						temp=sqrt(pow(Cyh[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(Cyh[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(Cyh[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
						if(min>temp){
							min=temp;
							min_n[0]=Cyh[0*(Nc/P)+j];
							min_n[1]=Cyh[1*(Nc/P)+j];
							min_n[2]=Cyh[2*(Nc/P)+j];
						}
					}else{
						break;
					}
				}

			}
		}
		// Xl Check
		if(done[4]){
			target = rank - ( (ko/k) * (mo/m) );
			if(target>0  && nbl>0){
				MPI_Wait(&reqs[14],&stats[8]);
				MPI_Wait(&reqs[15],&stats[9]);
				target=10000*n+100*yy+zz;
				for(j=indexOfBoxCxl[target];j<ic;j++){
					if(target==C[3*(Nc/P)+j]){
						temp=sqrt(pow(Cxl[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(Cxl[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(Cxl[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
						if(min>temp){
							min=temp;
							min_n[0]=Cxl[0*(Nc/P)+j];
							min_n[1]=Cxl[1*(Nc/P)+j];
							min_n[2]=Cxl[2*(Nc/P)+j];
						}
					}else{
						break;
					}
				}

			}
		}
		// Xh Check
		if(done[5]){
			target = rank + ( (ko/k) * (mo/m) );
			if(target<numtasks && nbh<1){
				MPI_Wait(&reqs[12],&stats[10]);
				MPI_Wait(&reqs[13],&stats[11]);
				target=10000*1+100*1+zz;
				for(j=indexOfBoxCxh[target];j<ic;j++){
					if(target==Cxh[3*(Nc/P)+j]){
						temp=sqrt(pow(Cxh[0*(Nc/P)+j]-Q[0*(Nq/P)+i],2)+pow(Cxh[1*(Nc/P)+j]-Q[1*(Nq/P)+i],2)+pow(Cxh[2*(Nc/P)+j]-Q[2*(Nq/P)+i],2));
						if(min>temp){
							min=temp;
							min_n[0]=Cxh[0*(Nc/P)+j];
							min_n[1]=Cxh[1*(Nc/P)+j];
							min_n[2]=Cxh[2*(Nc/P)+j];
						}
					}else{
						break;
					}
				}

			}
		}
		
		min_n[0]=0;
		min_n[1]=0;
		min_n[2]=0;
		mini=-1;
		min=1;
	}
}


void check_inc_C(){
	int s,z=0;
	int A=0;
	double a;
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
		// Jump to the next subtable when zeros
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
	quicksort(C,0,ic-1,L);
	// Fill indexOfBoxC table
	for(s=0;s<ic;s++){
		if(s>0){
			if(C[3*L+s]!=C[3*L+s-1]){
				indexOfBoxC[(int)C[3*L+s]]=s;
			}
		}else{
			indexOfBoxC[(int)C[3*L+s]]=0;
		}
	}
	
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

	indexOfBoxC=(int *) malloc(999999*sizeof(int));
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
	// if the number is ok for this process
		if( d[0] < nbh && d[1] < mbh && d[2] < kbh && d[0] >= nbl && d[1] >= mbl && d[2] >= kbl ){
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

	// if the number is ok for this process
		if( d[0] < nbh && d[1] < mbh && d[2] < kbh && d[0] >= nbl && d[1] >= mbl && d[2] >= kbl ){
			
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


	// quicksort(C,0,ic-1,Nc/P);
	// quicksort(Q,0,iq-1,Nq/P);
	
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
