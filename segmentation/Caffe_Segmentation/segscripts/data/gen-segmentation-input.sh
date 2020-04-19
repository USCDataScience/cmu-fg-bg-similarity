FILE_LIST=/install/cmu-fg-bg-similarity/ComputeFeatures/Features/CNN/ver2/fileList.txt
ln -s /images /install/cmu-fg-bg-similarity/segmentation/Caffe_Segmentation/segscripts/data/corpus

rm -rf ./ImgsList.txt
rm -rf ./ImgsList_IDL_input.txt

cp -R ${FILE_LIST} ./ImgsList.txt
for img in $( cat ./ImgsList.txt ); do 
	echo "data/corpus/${img} 1" >> ./ImgsList_IDL_input.txt
done

 
