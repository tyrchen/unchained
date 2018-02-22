GDRIVE=skicka
GDRIVE_DOWNLOAD=$(GDRIVE) download
SRC=/Notability/Blockchain
DST=.


sync:
	@$(GDRIVE_DOWNLOAD) $(SRC) $(DST)
