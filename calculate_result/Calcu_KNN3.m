function [ACC,F,P,R]= Calcu_KNN3(data,label,k,foldnumber)
    V=size(label,1);
    F=0;
    P=0;
    R=0;
    ACC=0;
    for i=1:foldnumber
        train_data=[data((1+(i-1)*foldnumber):(V/(2*foldnumber)+(i-1)*foldnumber),:);data((1+V/2+(i-1)*foldnumber):(V/(2*foldnumber)+V/2+(i-1)*foldnumber),:)];
        train_label=[label((1+(i-1)*foldnumber):(V/(2*foldnumber)+(i-1)*foldnumber),:);label((1+V/2+(i-1)*foldnumber):(V/(2*foldnumber)+V/2+(i-1)*foldnumber),:)];
        mdl = fitcknn(train_data,train_label,'NumNeighbors',k);%k为对应的1,2,3,4.....
        class = predict(mdl,data);
        [Fi,Pi,Ri] = compute_f(label,class);
        ACCi=Accuracy(label,class);
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