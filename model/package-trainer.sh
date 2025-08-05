rm -f capy-trainer.tar capy-trainer.tar.gz
tar cvf capy-trainer.tar package
gzip capy-trainer.tar
gsutil cp capy-trainer.tar.gz gs://capy-data/model/capy-trainer.tar.gz