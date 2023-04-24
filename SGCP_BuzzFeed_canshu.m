clear;
clc;
%% 加载数据集
load('BuzzFeed_new.mat');
V=182;
N=1417;
T=zeros(V,N,N);
for i=1:V
    for k=1:N
        if NU(i,k)==1
            for j=1:N
                if UU(j,k)==1
                    T(i,j,k)=1;
                end
            end               
        end
    end
end
%% 张量X,以及矩阵X1,X2,X3

X1 =tenmat(T,1);
X2 =tenmat(T,2);
X3 = tenmat(T,3);

% length=36;
length=81;

Result_SVM_ACC=zeros(length,10);
Result_SVM_F=zeros(length,10);
Result_SVM_P=zeros(length,10);
Result_SVM_R=zeros(length,10);

Result_KNN_ACC=zeros(length,10);
Result_KNN_F=zeros(length,10);
Result_KNN_P=zeros(length,10);
Result_KNN_R=zeros(length,10);

Result_LDA_ACC=zeros(length,10);
Result_LDA_F=zeros(length,10);
Result_LDA_P=zeros(length,10);
Result_LDA_R=zeros(length,10);

Result_Tree_ACC=zeros(length,10);
Result_Tree_F=zeros(length,10);
Result_Tree_P=zeros(length,10);
Result_Tree_R=zeros(length,10);

% value1 = [0.001, 0.01, 0.1, 1, 10, 100];
% value2 = [0.001, 0.01, 0.1, 1, 10, 100];

value1 = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 100];
value2 = [0.001, 0.01, 0.1, 0.05, 0.2, 0.5, 1, 10, 100];

for  jj = 1: 9
   for kk = 1: 9
      fprintf("jj= %f, kk=%f\n", jj,kk);
       lambda1=value1(jj);
       lambda2=value2(kk);
        for ii=1:10

r= 20; % 张量的cp秩
A = rand(V,r);
B = rand(N,r);
C = rand(N,r);

%% 参数设置
% lambda1=2;
% lambda2=0.001;
iter = 0;
tol = 10e-5;
max_iter = 200;
xi=10e-7;

obj_old=norm(double(X1)-A*khatrirao(C,B)','fro')^2;
tic
while iter < max_iter
    
        W=constructW(A, struct('NeighborMode', 'KNN', 'k', 5, 'WeightMode', 'HeatKernel', 't', 1));
        D=diag(sum(A,2));
        L=D-W;
        %L = speye(V) - NormAdjac(ConstructAffinity(A, 5));
        %% Update B

        CkhaA = khatrirao(C, A);
        B = B .* ((double(X2) * CkhaA) ./ (B * (CkhaA' * CkhaA) + xi));

        %% Update C

        BkhaA = khatrirao(B, A);
        C = C .* ((double(X3) * BkhaA) ./ (C * (BkhaA' * BkhaA) + xi));
        
        %% Update A
        
        CkhaB = khatrirao(C, B);
        A = A .* ((double(X1) * CkhaB+lambda2*W*A) ./ (A * (CkhaB' * CkhaB) + xi+lambda1+lambda2*D*A));
        
        
        obj=0.5*norm(double(X1)-A*CkhaB','fro')^2+lambda1*norm(A,1)+0.5*lambda2*trace(A'*L*A);
        err=abs((obj_old-obj)/obj);
        fprintf('obj=%f,err=%f\n',obj,err);
        if err < tol
            break;
        end
        iter = iter+1;
        obj_old=obj;
end
toc
U=full(A);
U=NormalizeData(U');
U=U';
[SVM_ACC,SVM_F,SVM_P,SVM_R]= Calcu_SVM3(U,label,foldnumber);
[KNN_ACC,KNN_F,KNN_P,KNN_R]= Calcu_KNN3(U,label,2,foldnumber);
[LDA_ACC,LDA_F,LDA_P,LDA_R]= Calcu_LDA3(U,label,foldnumber);
[Tree_ACC,Tree_F,Tree_P,Tree_R]= Calcu_Tree3(U,label,foldnumber);

Result_SVM_ACC(((jj-1)*9+kk),ii)=SVM_ACC;
Result_SVM_F(((jj-1)*9+kk),ii)=SVM_F;
Result_SVM_P(((jj-1)*9+kk),ii)=SVM_P;
Result_SVM_R(((jj-1)*9+kk),ii)=SVM_R;

Result_KNN_ACC(((jj-1)*9+kk),ii)=KNN_ACC;
Result_KNN_F(((jj-1)*9+kk),ii)=KNN_F;
Result_KNN_P(((jj-1)*9+kk),ii)=KNN_P;
Result_KNN_R(((jj-1)*9+kk),ii)=KNN_R;

Result_LDA_ACC(((jj-1)*9+kk),ii)=LDA_ACC;
Result_LDA_F(((jj-1)*9+kk),ii)=LDA_F;
Result_LDA_P(((jj-1)*9+kk),ii)=LDA_P;
Result_LDA_R(((jj-1)*9+kk),ii)=LDA_R;

Result_Tree_ACC(((jj-1)*9+kk),ii)=Tree_ACC;
Result_Tree_F(((jj-1)*9+kk),ii)=Tree_F;
Result_Tree_P(((jj-1)*9+kk),ii)=Tree_P;
Result_Tree_R(((jj-1)*9+kk),ii)=Tree_R;


        end
   end
end

