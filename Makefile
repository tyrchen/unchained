GDRIVE=skicka
GDRIVE_DOWNLOAD=$(GDRIVE) download
BASE_SRC=/Notability
BASE_DST=.

WP=$(BASE_SRC)/Blockchain
WP_DST=$(BASE_DST)/blockchain

TALKS=$(BASE_SRC)/talks
TALKS_DST=$(BASE_DST)/talks


sync:
	@$(GDRIVE_DOWNLOAD) $(WP) $(WP_DST)
	@$(GDRIVE_DOWNLOAD) $(TALKS) $(TALKS_DST)
