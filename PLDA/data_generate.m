
% [B, I]= maxk(A,1,1);
% 
% sz = [6906 2];
% varTypes = ["string","string","string","string","string","string","string","string","string","string"];
% varNames = ["1","2","3","4","5","6","7","8","9","10"];
% temps = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);
% for i = 1:160
%     for j = 1:10
%         temps(i, j) = vocab(I(i,j),1);
%     end
% end


[B, I]= maxk(A,20,2);

sz = [246 20];
varTypes = ["string","string","string","string","string","string","string","string","string","string","string","string","string","string","string","string","string","string","string","string"];
varNames = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"];
temps = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);
for i = 1:246
    for j = 1:20
        temps(i, j) = vocab(I(i,j),1);
    end
end
