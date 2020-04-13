for i in {1..10571}; do
  if [ ! -f /srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/features/sift/corpusNsifts/$i.ppm.hesaff.sift ]; then
    echo $i
    ./haff_cvpr09 /srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/features/sift/corpusNsifts/$i.ppm
  fi
done
