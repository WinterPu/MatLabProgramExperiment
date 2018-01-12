W = [0.2491,0.6614,0.0892];
Ep = imread('C:\Users\Winter_Pu\Desktop\NN\Image_and_ImageData\Eprime.png');
A = imread('C:\Users\Winter_Pu\Desktop\NN\Image_and_ImageData\key1.png');
B = imread('C:\Users\Winter_Pu\Desktop\NN\Image_and_ImageData\key2.png');
I=double(zeros(300,400));
for i =1:300
    for j = 1:400;
    I(i,j) = double((Ep(i,j) - (A(i,j)*W(1,1)) -(B(i,j)*W(1,2)))/W(1,3));
    end
end
imshow(I);