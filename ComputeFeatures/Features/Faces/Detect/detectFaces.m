function detectFaces(inputDir, outputDir)
% DETECTFACES takes a directory with images, detect and store in the outputDir
addpath(genpath('../../../utils'))

files = getAllFiles(inputDir);
faceDetector = vision.CascadeObjectDetector();
if ~exist(outputDir, 'dir')
    mkdir(outputDir)
end

fprintf('Found %d files..\n', numel(files));
for i = 1 : numel(files)
    I = imread(fullfile(inputDir, files{i}));
    bboxes = step(faceDetector, I);
    [rpath, fname, fext] = fileparts(files{i});
    for j = 1 : size(bboxes, 1)
        x = bboxes(j, 1); y = bboxes(j, 2);
        wd = bboxes(j, 3); ht = bboxes(j, 4);
        face = I(y : y + ht, x : x + wd, :);
        imwrite(face, fullfile(outputDir, rpath, [fname, '_', num2str(j), fext]));
    end
    fprintf('Done for %s (%d/%d)\n', files{i}, i, numel(files));
end

