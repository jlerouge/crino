IMAGES=concat.png sequential.png 
.SUFFIXES : .dia .png

all: ${IMAGES}

.dia.png:
	dia --export=$*.png $*.dia

clean:
	rm -f ${IMAGES}
