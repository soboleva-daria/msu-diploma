for i in $( ls | grep .npy ); 
do
	mv $i pickle
done
