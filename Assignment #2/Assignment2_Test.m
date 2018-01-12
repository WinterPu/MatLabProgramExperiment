test_input = importdata('C:\Users\Winter_Pu\Desktop\NN\Data\in_test.txt');
test_output = importdata('C:\Users\Winter_Pu\Desktop\NN\Data\out_test.txt');
[mi_test,ni_test] = size(test_input);
[mo_test,no_test] = size(test_output);

Confusion_Matrix = zeros(10,10);

%---Normalize the test input---
for i= 1:mi_test
    count_test =0;
    x_test=1;
    y_test=2;
  while(count_test<8)
    
    % test_temp_normal_result = (sqrt(test_input(i,x_test)^2 + test_input(i,y_test)^2)); 
    %The above statement is wrong, it will distort the shape of the handwritten pattern
    test_temp_normal_result = 100;
    if(test_temp_normal_result == 0)
            test_input(i,x_test) = 0;    
            test_input(i,x_test) = 0;   
        else     
            test_input(i,x_test) = test_input(i,x_test)/test_temp_normal_result;
            test_input(i,y_test)=test_input(i,y_test)/test_temp_normal_result;
     end
    x_test = x_test+2;
    y_test = y_test+2;
    count_test=count_test+1;
  end
end


num_layer = 3;
%Randomly initialize W and bias
for i=1:num_layer
    if(i == 1)
    a_cell_test{i} = zeros(16,1);
     else    
    a_cell_test{i} = zeros(10,1);
    end
end

 a_cell_test{num_layer+1} = zeros(10,1);

count_correct = 0;

for q = 1:mi_test
    
    
    test_irow = double(test_input(q,:)');
    a_cell_test{1} = test_irow;
    
   for layer = 1:num_layer
    % a_cell_test{layer+1} = W_backup{layer} * a_cell_test{layer} +bias_backup{layer};
     a_cell_test{layer+1} = W{layer} * a_cell_test{layer} +bias{layer};
        for k = 1:10
         a_cell_test{layer+1}(k,1) = Activation_func(a_cell_test{layer+1}(k,1));
        end
    end
    
    t_test = zeros(10,1);
    temp_output  = test_output(q,:);
    t_test(temp_output+1,1) = 1; 
    
    %Get the max value's position in the matrix 
    %set it to 1 and others to 0
    a_result = zeros(10,1);
    temp_a_cell_result = a_cell_test{Num_Multilayer+1};
    position =0;
    max_value =0;
    for i = 1:10
        if( temp_a_cell_result(i,1)>max_value)
            max_value = temp_a_cell_result(i,1);
            position = i;
        end
    end
    a_result(position,1) = 1;
    
    
    %--for Confusion_Matrix----
    for j  = 1:10
        if(t_test(j,1) == 1)
            t_test_position = j;
        end
    end
    Confusion_Matrix(t_test_position,position) = Confusion_Matrix(t_test_position,position) +1;
    %---------------------------
    
    Error_test =(norm(t_test - a_cell_test{Num_Multilayer+1}))^2;
    if(a_result == t_test )
        count_correct = count_correct +1;
    end
end
    
display(Confusion_Matrix);
display(count_correct);
Precision = count_correct / mi_test * 100;
display(Precision);


%Export W and bias----------------------
flag_backup_or_not = 0;

if(flag_backup_or_not == 0)
    file_W_ID = fopen('W_file.txt','w');
    file_bias_ID = fopen('bias_file.txt','w');
    formatSpec = '%s';
    [filerows_W,filecols_W] = size(W);
    [filerows_bias,filecols_bias] = size(bias);
    
for row_file_W = 1:filerows_W
       fprintf(file_W_ID,formatSpec,W{row_file_W,:});
       fprintf(file_W_ID,formatSpec,'\n');
end
    
for     row_file_bias = 1:filerows_bias
        fprintf(file_bias_ID,formatSpec,bias{row_file_bias,:});
        fprintf(file_bias_ID,formatSpec,'\n');
end
    fclose(file_W_ID);
    fclose(file_bias_ID);
else 
    file_W_ID = fopen('W_file.txt','w');
    file_bias_ID = fopen('bias_file.txt','w');
    formatSpec = '%s';
    [filerows_W,filecols_W] = size(W_backup);
    [filerows_bias,filecols_bias] = size(bias_backup);
for row_file_W = 1:filerows_W
    fprintf(file_W_ID,formatSpec,W_backup{row_file_W,:});
    fprintf(file_W_ID,formatSpec,'\n');
end
for row_file_bias = 1:filerows_bias
    fprintf(file_bias_ID,formatSpec,bias_backup{row_file_bias,:});
    fprintf(file_bias_ID,formatSpec,'\n');
end
fclose(file_W_ID);
fclose(file_bias_ID);

end
%-----------------------------------------------
