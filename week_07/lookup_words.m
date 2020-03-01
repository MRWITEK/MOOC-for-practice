function word_indices = lookup_words(vocabList, strings_list)
word_indices = [];

% vocabList is cell array of strings
% vocabList{i} = str;

% use built-in associative array for lookup
lookup_list = {};
for i = 1:length(vocabList)
    lookup_list.(vocabList{i}) = i;
end

for i = 1:length(strings_list)
    str = strings_list{i};
    try
        word_indices = [word_indices; lookup_list.(str)];
    catch % error on failed lookup
        % we skip the word
    end
end
end
