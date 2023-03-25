clear,clc;

% 2021/01/06 finish
I=cell(1,200); %store image information
K=cell(1,200); %store image information
L=cell(1,200); %store image information
total=cell(1,200);%store image information
Y(200)=0;% Output
Y(101:200)=1; % label 0 img(1-100),and label 1 img(101-200).
Y=Y';%trans
M(2,2)=0;
filter1=[1 1 -1;
         -1 0 1;
        -1 1 1];
% filter2=[1 1 -1;
%          1 0 -1;
%         1 -1 -1];
filter2=[1 1 -1;
         1 0 -1;
        1 -1 -1];
for b=1:200
   m1=imread(['D:/train1000/',int2str(b),'.png']); 
   I{b}=imresize(m1,[8 8]); %I{1}。。。I{200} every img
   total{b}=m1;%store
   I{b}=cat(3,I{b},I{b});%轉算成2 feature map
   %convolution
   I{b}(:,:,1)=imfilter(I{b}(:,:,1),filter1);
   I{b}(:,:,2)=imfilter(I{b}(:,:,2),filter2);
   %uint no minus number,so is zeros.
   K{b}=I{b}(:,:,1);
   L{b}=I{b}(:,:,2);
   
   fun=@(block_struct)max_matrix(block_struct.data);%Maxpooling 2x2  
   K{b}=blockproc(K{b},[2 2],fun); %Seperaetely pooling feature1
   L{b}=blockproc(L{b},[2 2],fun); %Seperaetely pooling feature2
   
   I{b}=cat(3,K{b},L{b});%轉算成2 feature map
   K{b}=imfilter(K{b},filter1); %convolution
   L{b}=imfilter(L{b},filter2); %convolution
   %===============================================================%
   K{b}=blockproc(K{b},[2 2],fun); %Seperaetely pooling feature1
   L{b}=blockproc(L{b},[2 2],fun); %Seperaetely pooling feature2
   
   I{b}=cat(3,K{b},L{b});% merge feature become result

   I{b}=reshape(I{b},8,1); %flatten 
   I{b}(9,1)=1;
   X=[I{:}];% Cell transfer to matrix 9x200 ;01/06 make
   X=double(X');

%max2 blkproc A = inv(X'*X)*(X'*Y) or X\Y is least square function    
end

 A= X\Y; %  A= inv(X'*X)*(X'*Y); least square
 Yp=X*A;
 aa=0;cc=0;
 bb=0;dd=0;
 char(1,200)=0;
 label(1,200)=0;
% Judge '0' or '1'
for i=1:200   
 if i<101
  if Yp(i)<0.5
     aa=aa+1;
     char(i)=0;% true
     label(i)=i;
  else 
     bb=bb+1;
     char(i)=1;% '0'Recognize to '1'　　
     label(i)=i;
  end
else    
 if Yp(i)>0.5
     cc=cc+1; % true is 1
     char(i)=1;
     label(i)=i;
 else 
     dd=dd+1; % 1 recognize to '0'
     char(i)=0;
     label(i)=i;
  end
 end
end
M(1,1)=aa;M(2,1)=bb;
M(2,2)=cc;M(1,2)=dd;

final=cell(1,15); %store image information
% label2=cell(1,15);%store label2 information

%randly choose 15 img
 for i=1:15
   rr=randi(200,1);
   final{i}=total{rr};
   labels(i)=label(rr);
   label2(i)=char(rr);
   true(i)=Y(rr);
 end
 
 
 figure()
 h=heatmap(M); % confusion Matrix
        
 figure()
 for j=1:15
      if true(j)==label2(j)
      subplot(3,5,j);
      imshow(final{j});
      txta=num2str(label2(j)); txtb='(';
      
      txtc=num2str(labels(j)); txtd=')';
      
      str = strcat(txta,txtb,txtc,txtd);
      title(str,'Color','blue');  % R is red
      else
      subplot(3,5,j);
      imshow(final{j});
      txta=num2str(label2(j));
      txtb='(';
      txtc=num2str(labels(j));
      txtd=')';
      str = strcat(txta,txtb,txtc,txtd);
      title(str,'Color','red');  % R is red         
      end
 end

