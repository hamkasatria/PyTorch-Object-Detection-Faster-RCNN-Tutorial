SHELL=/bin/bash

# MAKEFILE to quickly execute utility commands
#
# You can use tab for autocomplete in your terminal
# > make[space][tab]
#

format:
	@echo "Running isort..."
	@echo $(shell pwd)
	@isort $(shell pwd) --profile "black"
	@echo "Running black..."
	@black $(shell pwd)

check-format:
	@echo "Running check isort..."
	@isort --check $(shell pwd) || true
	@echo "Running check flake8..."
	@flake8 $(shell pwd) --ignore=E501 || true

# ---------------------//----------------

run-realtime:
	@python realtime.py
	
run-training:
	@ipython -c "%run training_script.ipynb"

run-test-backbone:
	@python tests/test_backbone_resnet.py

run-test-framework:
	@python tests/test_faster_RCNN.py

run-test:
	@make run-test-backbone
	@make run-test-framework
	
init-requirements-all:
	@make create-env
	@pip install -r requirements.txt
	@pip install -r requirements-dev.txt
	@make run-setup
	
run-setup:
	@python setup.py install

craete-env:
	@conda create --name faster-rcnn-tutorial -y
	@conda activate faster-rcnn-tutorial
	@conda install python=3.8 -y
	@python3 -m venv faster-rcnn-tutorial
	@source faster-rcnn-tutorial/bin/activate

