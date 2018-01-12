clear
% use [clear]command  so that we can use cell without error

input = importdata('C:\Users\Winter_Pu\Desktop\NN\Data\in_train.txt');
output = importdata('C:\Users\Winter_Pu\Desktop\NN\Data\out_train.txt');
[mi,ni]=size(input);
[mo,no]=size(output);

%Normalize the input
for i= 1:mi
    count =0;
    x=1;
    y=2;
    
    while(count<8)
    %normal_result = sqrt(input(i,x)^2 + input(i,y)^2);
    %The above statement is wrong, it will distort the shape of the handwritten pattern
    normal_result = 100;
    
    if(normal_result == 0)  %Take care the condition  normal_result is zero
      input(i,x) =0;
      input(i,y)=0;
    else   
      input(i,x) = input(i,x)/normal_result ;
      input(i,y)=input(i,y)/normal_result;
    end
    
    x = x+2;
    y = y+2;
    count=count+1;
    end
end
%-----------------------------------------------------
Epoch =1;
Error =0;

Num_Multilayer = 3;

%Randomly initialize W and bias
Randomly_Initialize_Rate = 0.1;
for i=1:Num_Multilayer
if(i == 1)
    W{i} = Randomly_Initialize_Rate * rand(10,16);
    deltaW{i} = zeros(10,16);
    a_cell{i} = zeros(16,1);
    Momentum_delta_W{i} = zeros(10,16);
else
    W{i} = Randomly_Initialize_Rate * rand(10,10);
    deltaW{i} = zeros(10,10);
    a_cell{i} = zeros(10,1);
    Momentum_delta_W{i} =  zeros(10,10);
end
    bias{i} = Randomly_Initialize_Rate * rand(10,1);
    sigma{i} = zeros(10,1);
    delta_bias{i} = zeros(10,1);
    Momentum_delta_bias{i} = zeros(10,1);
end
 a_cell{Num_Multilayer+1} = zeros(10,1);
 
learning_rate = 0.1;
MaxEpochLimit = 3000;
ErrorLimit = 0;
n_L = zeros(10,1);


%---Momentum Method
gamma = 0.05;
%----------------------------

% for backup good performance W bias
backupRange = 2000;
backupflag =0;
%-------------------------

while(Epoch ==1|| (Epoch <MaxEpochLimit && Error > ErrorLimit))
    Error = 0;
for q = 1:mi
    
    i_row = double(input(q,:)'); %1*16 -> 16*1
    a_cell{1} = i_row;
    
    for layer = 1:Num_Multilayer
     a_cell{layer+1} = W{layer} * a_cell{layer} +bias{layer};
          
        for k = 1:10
         a_cell{layer+1}(k,1) = Activation_func(a_cell{layer+1}(k,1));
         %a_cell{layer+1}(k,1) = logsig(a_cell{layer+1}(k,1));
        end
    end
    n_L =  a_cell{Num_Multilayer+1};
    
%        if(q == 1)
%        display(n_L);
%        end

    t = zeros(10,1);
    temp_output  = output(q,:);
    t(temp_output+1,1) = 1; 
    
    Error = Error+(norm(t - a_cell{Num_Multilayer+1}))^2;
      
%      if(q==20)
%           display(t - n_L);
%      end

    nL_fdev_maxtrix = zeros(10,10);
    for p = 1:10
        nL_fdev_maxtrix(p,p) = Activation_func_drev(n_L(p,1));
    end
    
    %Backpropagate the error
    %for layer = M
    sigma{Num_Multilayer}= (-2) * nL_fdev_maxtrix * (t - a_cell{Num_Multilayer+1});
    deltaW{Num_Multilayer} = sigma{Num_Multilayer}* (a_cell{Num_Multilayer})';
    delta_bias{Num_Multilayer} = sigma{Num_Multilayer};
    
    
    %for layer < M
    layer =Num_Multilayer-1;
while(layer >=1)
    
    n_L = a_cell{layer+1};
    nL_fdev_maxtrix = zeros(10,1);
    for p = 1:10
        nL_fdev_maxtrix(p,p) = Activation_func_drev(n_L(p,1));
    end
    
    
    sigma{layer} = nL_fdev_maxtrix * ( W{layer+1} * sigma{layer+1});
        
    deltaW{layer} = sigma{layer}* (a_cell{layer})';
    delta_bias{layer} = sigma{layer};
%for layer = 4 ---- sigma(10*1) * a_cell(1*10)
%for layer = 1 ---- sigma(10*1) * a_cell(1*16)
    layer = layer -1;
end




%Calculate weight and bias updates
    layer = Num_Multilayer;
while(layer >=1)
    W{layer} =  W{layer} - learning_rate* deltaW{layer} - gamma * Momentum_delta_W{layer};
    bias{layer} = bias{layer} - learning_rate* delta_bias{layer}- gamma* Momentum_delta_bias{layer};
    layer = layer -1;
end

     Momentum_delta_W = deltaW;
     Momentum_delta_bias = delta_bias;
     
end

    Epoch = Epoch +1;
     display(Error);
     display(Epoch);
     
    %---Plot the Error-Vs-Epoch----------- 
     hold on;
     plot(Epoch,Error,'r:diamond');
     drawnow;
     axis([0,1000,0,6000]);
     grid on;
     pause(0.1);
     %------------------------
     % Back up W and bias
     if(Error <  backupRange)
        backupflag = 1;
        W_backup = W;
        bias_backup = bias;
        backupRange = Error;
     end
     
end

