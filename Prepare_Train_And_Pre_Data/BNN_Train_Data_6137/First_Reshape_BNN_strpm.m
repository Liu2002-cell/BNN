%% 扁平化建筑结构参数，方便后续读取结构参数

srcDir = 'D:\ReprBldID_strpm';
dstDir = 'D:\BNN_Bld_strpm';
if ~exist(dstDir,'dir'), mkdir(dstDir); end

splitCSV    = @(ln) regexp(char(ln), '"[^"]*"|[^,]+', 'match');
stripQuotes = @(c) regexprep(c, '^"(.*)"$', '$1');

% 要复制/扁平化的次数
repeatN = 52;

files = dir(fullfile(srcDir,'ReprBldID_strpm_*.csv'));
for k = 1:numel(files)

    lines = readlines(fullfile(srcDir, files(k).name));

    %% ——— 1) 拆前三行并去引号 ———
    raw1 = splitCSV(lines(1)); raw1 = cellfun(stripQuotes, raw1,'Uni',false);
    raw2 = splitCSV(lines(2)); raw2 = cellfun(stripQuotes, raw2,'Uni',false);
    raw3 = splitCSV(lines(3)); raw3 = cellfun(stripQuotes, raw3,'Uni',false);

    %% ——— 2) 可选：替换“第一行 第3列” ———
    if numel(raw1) >= 3
        raw1{3} = 'Hysteretic parameter tao';
    end

    %% ——— 3) 裁剪到最后一个非空列 ———
    isEmpty = @(c) isempty(c) || all(isspace(c));
    last1 = find(~cellfun(isEmpty, raw1),1,'last');
    last2 = find(~cellfun(isEmpty, raw2),1,'last');
    last3 = find(~cellfun(isEmpty, raw3),1,'last');
    row1 = raw1(1:last1);
    row2 = raw2(1:last2);
    row3 = raw3(1:last3);

    %% ——— 4) 只取第4行起的前52行，拆分并填充空单元格为 '0' ———
    startLine = 4;
    endLine = min(startLine + repeatN - 1, numel(lines));
    dataLines = lines(startLine:endLine);
    nRows    = numel(dataLines);
    dataRows = cell(nRows, last3);
    for i = 1:nRows
        toks = splitCSV(dataLines(i));
        toks = cellfun(stripQuotes, toks,'Uni',false);
        toks = toks(1:min(end,last3));
        if numel(toks) < last3
            toks(end+1:last3) = {''};
        end
        dataRows(i,:) = toks;
    end
    emptyMask = cellfun(@(c) isempty(c) || all(isspace(c)), dataRows);
    dataRows(emptyMask) = {'0'};

    %% ——— 5) 构造“复制52次并扁平化”的第三行头（跳过第一列） ———
    % 前三行第一列不参与扁平化
    prefixRow = row1(1);
    flatHeaders = cell(1, (last3-1) * repeatN);
    for rep = 1:repeatN
        for j = 2:last3
            flatHeaders{ (rep-1)*(last3-1) + (j-1) } = sprintf('%d_%s', rep, row3{j});
        end
    end

    %% ——— 6) 新表头 & 数据扁平化（跳过第一列） ———
    headerNew = [ prefixRow, row1(2:last1), flatHeaders ];
    % 扁平化这52行数据，从第二列开始
    dataSub = dataRows(:,2:last3); 
    dataFlat = reshape(dataSub.', 1, []); 
    % 若实际行少于52，用0补齐剩余行的数据
    if nRows < repeatN
        padCount = (repeatN - nRows) * (last3-1);
        dataFlat = [dataFlat, repmat({'0'}, 1, padCount)];
    end
    dataNew = [ row2(1), row2(2:last2), dataFlat ];

    %% ——— 7) 输出CSV ———
    idx     = regexp(files(k).name, 'ReprBldID_strpm_(\d+)', 'tokens','once');
    outName = sprintf('BNN_Bld_strpm_%s.csv', idx{1});
    fid     = fopen(fullfile(dstDir,outName), 'w');

    % 写表头
    fprintf(fid, '%s', headerNew{1});
    for j = 2:numel(headerNew)
        fprintf(fid, ',%s', headerNew{j});
    end
    fprintf(fid, '\n');

    % 写扁平化后的一行数据
    fprintf(fid, '%s', dataNew{1});
    for j = 2:numel(dataNew)
        fprintf(fid, ',%s', dataNew{j});
    end
    fprintf(fid, '\n');

    fclose(fid);
    fprintf('→ %s done\n', outName);
end

disp('All files processed.');
