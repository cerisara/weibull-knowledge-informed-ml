rsync -av lully:github/weibull-knowledge-informed-ml/xtof/lightning_logs ./
tensorboard --logdir ./lightning_logs/

exit

convert -density 300 -background black /home/xtof/Downloads/equiloss.svg eq.jpg
