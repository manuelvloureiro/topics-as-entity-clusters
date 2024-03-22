################################################################################
# GLOBALS
################################################################################

PYTHON_INTERPRETER = python
CORPORA = data/corpora
INTERIM = data/interim

QRANK_URL = https://qrank.wmcloud.org/download/qrank.csv.gz

WIKIPEDIA_FOLDER = $(CORPORA)/wikipedia
WIKIPEDIA_URL = https://dumps.wikimedia.org/enwiki/latest
WIKIPEDIA = enwiki-latest-pages-articles.xml.bz2

WIKIDATA_FOLDER = $(CORPORA)/wikidata
WIKIDATA_URL = https://dumps.wikimedia.org/wikidatawiki/entities
WIKIDATA = latest-all.json.bz2

LANGUAGES = en fr de pl ru tr es pt ms ar it th
SPACY_EN = en_core_web_sm-3.0.0
SPACY_FR = fr_core_news_sm-3.0.0
SPACY_DE = de_core_news_sm-3.0.0
SPACY_PL = pl_core_news_sm-3.0.0
SPACY_RU = ru_core_news_sm-3.0.0
SPACY_ES = es_core_news_sm-3.0.0
SPACY_PT = pt_core_news_sm-3.0.0
SPACY_IT = it_core_news_sm-3.0.0

################################################################################
# FUNCTIONS
################################################################################

define log
	@$(PYTHON_INTERPRETER) -c "from topic_inference.utils import console_log;console_log('$(1)', 'make $(@)')"
endef

define spacy_lang
	wget https://github.com/explosion/spacy-models/releases/download/$(1)/$(1).tar.gz
	$(PYTHON_INTERPRETER) -m pip install ./$(1).tar.gz
	rm ./$(1).tar.gz
endef

################################################################################
# TASKS
################################################################################

.PHONY: qrank
qrank:
	wget -O $(WIKIDATA_FOLDER)/qrank.csv.gz $(QRANK_URL)
	gzip -d $(WIKIDATA_FOLDER)/qrank.csv.gz


.PHONY: requirements
requirements:
	pip install --upgrade pip
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel jupyter
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

.PHONY: spacy_resources
spacy_resources:
	$(call spacy_lang,$(SPACY_EN))
	$(call spacy_lang,$(SPACY_FR))
	$(call spacy_lang,$(SPACY_DE))
	$(call spacy_lang,$(SPACY_PL))
	$(call spacy_lang,$(SPACY_RU))
	$(call spacy_lang,$(SPACY_ES))
	$(call spacy_lang,$(SPACY_PT))
	$(call spacy_lang,$(SPACY_IT))

.PHONY: wikidata
wikidata: wikidata-download wikidata-summary wikidata-counts

.PHONY: wikidata-download
wikidata-download:
	$(call log,Starting download: $(WIKIDATA_FOLDER)/$(WIKIDATA)_temp)
	mkdir -p -- $(WIKIDATA_FOLDER)
	rm -f $(WIKIDATA_FOLDER)/$(WIKIDATA)_temp  # remove file in case it exists
	# download wikidata dump
	wget -O $(WIKIDATA_FOLDER)/$(WIKIDATA)_temp \
			$(WIKIDATA_URL)/$(WIKIDATA)
	mv $(WIKIDATA_FOLDER)/$(WIKIDATA)_temp $(WIKIDATA_FOLDER)/$(WIKIDATA)
	$(call log,Completed download: $(WIKIDATA_FOLDER)/$(WIKIDATA))
	# record the timestamp of the last dump
	wget -O- -q $(WIKIDATA_URL) | \
		grep $(WIKIDATA) | \
		awk {'print $$3 " " $$4'} | \
		head -n1 > $(WIKIDATA_FOLDER)/dump-timestamp
	$(call log,Saved Wikidata timestamp: $(shell cat $(WIKIDATA_FOLDER)/dump-timestamp))

.PHONY: wikidata-summary
wikidata-summary:
	$(call log,Running wikidata summary generation)
	$(PYTHON_INTERPRETER) paper_experiments/wikidata/make_wikidata_summary.py \
			$(WIKIDATA_FOLDER)/$(WIKIDATA) \
			$(WIKIDATA_FOLDER)/wikidata_summary.jsonl_temp \
			--languages $(LANGUAGES) --verbose
	mv $(WIKIDATA_FOLDER)/wikidata_summary.jsonl_temp $(WIKIDATA_FOLDER)/wikidata_summary.jsonl
	$(call log,Completed summary: $(WIKIDATA_FOLDER)/wikidata_summary.jsonl)

.PHONY: wikidata-counts
wikidata-counts:
	$(call log,Running wikidata property counts)
	$(call log,Creating backup of wikidatamatcher folder)
	mkdir -p -- $(INTERIM)/wikidatamatcher
	cp -r $(INTERIM)/wikidatamatcher $(INTERIM)/wikidatamatcher_backup
	$(call log,Creating wikidata properties csvs)
	$(PYTHON_INTERPRETER) paper_experiments/wikidata/make_wikidata_property_csv.py \
			$(WIKIDATA_FOLDER)/wikidata_summary.jsonl \
			$(INTERIM)/wikidatamatcher/ \
			--properties instance_of facet_of subclass_of --verbose
	$(call log,Completed running wikidata property counts)
