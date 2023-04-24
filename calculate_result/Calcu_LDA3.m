function [ACC,F,P,R]= Calcu_LDA3(data,label,foldnumber)
    V=size(label,1);
    F=0;
    P=0;
    R=0;
    ACC=0;
    for i=1:foldnumber
        train_data=[data((1+(i-1)*foldnumber):(V/(2*foldnumber)+(i-1)*foldnumber),:);data((1+V/2+(i-1)*foldnumber):(V/(2*foldnumber)+V/2+(i-1)*foldnumber),:)];
        train_label=[label((1+(i-1)*foldnumber):(V/(2*foldnumber)+(i-1)*foldnumber),:);label((1+V/2+(i-1)*foldnumber):(V/(2*foldnumber)+V/2+(i-1)*foldnumber),:)];
        lda = fitcdiscr(train_data,train_label,'DiscrimType','pseudoLinear');
        ldaclass = predict(lda,data);
        [Fi,Pi,Ri] = compute_f(label,ldaclass);
        ACCi=Accuracy(label,ldaclass);
        F=F+Fi;
        P=P+Pi;
        R=R+Ri;
        ACC=ACC+ACCi;
    end
    F=F/foldnumber;
    P=P/foldnumber;
    R=R/foldnumber;
    ACC=ACC/foldnumber;
end