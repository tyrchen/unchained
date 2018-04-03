GDRIVE=skicka
GDRIVE_DOWNLOAD=$(GDRIVE) download
BASE_SRC=/Notability
BASE_DST=.

WP=$(BASE_SRC)/Blockchain
WP_DST=$(BASE_DST)/blockchain

TALKS=$(BASE_SRC)/talks
TALKS_DST=$(BASE_DST)/talks

VERSION=$(strip $(shell cat version))
RELEASE_VERSION=v$(VERSION)
GIT_BRANCH=$(strip $(shell git symbolic-ref --short HEAD))
GIT_VERSION="$(strip $(shell git rev-parse --short HEAD))"
GIT_LOG=$(shell git log `git describe --tags --abbrev=0`..HEAD --pretty="tformat:%h | %s [%an]\n" | sed "s/\"/'/g")
RELEASE_BODY=release on branch __$(GIT_BRANCH)__\n\n$(GIT_LOG)
RELEASE_DATA='{"tag_name": "$(RELEASE_VERSION)", "name": "$(RELEASE_VERSION)", "target_commitish": "master", "body": "$(RELEASE_BODY)"}'
RELEASE_URL=https://api.github.com/repos/tyrchen/unchained/releases

sync:
	@$(GDRIVE_DOWNLOAD) $(WP) $(WP_DST)
	@$(GDRIVE_DOWNLOAD) $(TALKS) $(TALKS_DST)


release:
ifeq ($(GITHUB_TOKEN),)
	@echo "To generate a release, you need to define 'GITHUB_TOKEN' in your env."
else
	@echo "Create a release on $(RELEASE_VERSION)"
	@git tag -a $(RELEASE_VERSION) -m "Release $(RELEASE_VERSION). Revision is: $(GIT_VERSION)"
	@git push origin $(RELEASE_VERSION)
	curl -s -d $(RELEASE_DATA) "$(RELEASE_URL)?access_token=$(GITHUB_TOKEN)"
endif

delete-release:
	@echo "Delete a release on $(RELEASE_VERSION)"
	@git tag -d $(RELEASE_VERSION) | true
	@git push -f -d origin $(RELEASE_VERSION) | true
