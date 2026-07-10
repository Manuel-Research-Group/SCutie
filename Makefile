SHELL := /bin/bash
COLMAP_RECON_DIR = /home/newton/forks/Colmap/workspaces
WORKSPACES_ROOT = workspaces

.PHONY: publish-workspace-to-colmap clone-workspace start

publish-workspace-to-colmap:
	@# 1. Validação de Argumentos
	@if [ -z "$(path)" ] || [ -z "$(name)" ]; then \
		echo "⚠️  Uso incorreto."; \
		echo "   Execute: make publish-workspace-to-colmap path=pasta_origem name=nome_destino"; \
		exit 1; \
	fi

	@# 2. Definição dos caminhos completos
	# Aqui concatenamos 'workspaces/' com o nome que você passou
	$(eval SOURCE_DIR := $(WORKSPACES_ROOT)/$(path))
	$(eval TARGET_DIR := $(COLMAP_RECON_DIR)/$(name))

	@# 3. Verificação: A pasta de origem existe?
	@if [ ! -d "$(SOURCE_DIR)" ]; then \
		echo "❌ ERRO: A pasta de origem não foi encontrada:"; \
		echo "   -> $(SOURCE_DIR)"; \
		echo "   Verifique se o nome está correto dentro da pasta workspaces."; \
		exit 1; \
	fi

	@# 3.5. Verificação: A escala foi configurada?
	@if [ ! -f "$(SOURCE_DIR)/workspace_config.json" ]; then \
		echo "❌ ERRO: Escala não definida."; \
		echo "   Configure a escala no SCutie antes de publicar."; \
		echo "   (Menu: Workspace > Configurar Escala...)"; \
		exit 1; \
	fi

	@# 4. Verificação: O destino já existe? (Proteção contra overwrite)
	@if [ -d "$(TARGET_DIR)" ]; then \
		echo "❌ ERRO: Já existe um projeto com esse nome no destino:"; \
		echo "   -> $(TARGET_DIR)"; \
		echo "   Escolha outro 'name' ou apague a pasta de destino manualmente."; \
		exit 1; \
	fi

	@# 5. Execução da Cópia
	@echo "📦 Publicando workspace..."
	@echo "   📂 Origem: $(SOURCE_DIR)"
	@echo "   📂 Destino: $(TARGET_DIR)"
	@mkdir -p "$(COLMAP_RECON_DIR)"
	@cp -r "$(SOURCE_DIR)" "$(TARGET_DIR)"
	@echo "✅ Sucesso! Dados copiados para reconstructions."

clone-workspace:
	@# 1. Validação de Argumentos
	@if [ -z "$(src)" ] || [ -z "$(dest)" ]; then \
		echo "⚠️  Uso incorreto."; \
		echo "   Execute: make clone-workspace src=workspace_original dest=workspace_clonado"; \
		exit 1; \
	fi

	@# 2. Definição dos caminhos completos
	# Aqui manipulamos caminhos estritamente dentro de WORKSPACES_ROOT
	$(eval SOURCE_DIR := $(WORKSPACES_ROOT)/$(src))
	$(eval TARGET_DIR := $(WORKSPACES_ROOT)/$(dest))

	@# 3. Verificação: A pasta de origem existe?
	@if [ ! -d "$(SOURCE_DIR)" ]; then \
		echo "❌ ERRO: O workspace de origem não foi encontrado:"; \
		echo "   -> $(SOURCE_DIR)"; \
		echo "   Verifique se o nome está correto na pasta de workspaces."; \
		exit 1; \
	fi

	@# 4. Verificação: O destino já existe? (Proteção contra overwrite)
	@if [ -d "$(TARGET_DIR)" ]; then \
		echo "❌ ERRO: Já existe um workspace com o nome de destino:"; \
		echo "   -> $(TARGET_DIR)"; \
		echo "   Escolha outro 'dest' ou apague a pasta existente manualmente."; \
		exit 1; \
	fi

	@# 5. Execução da Cópia
	@echo "🔄 Clonando workspace..."
	@echo "   📂 Origem:  $(SOURCE_DIR)"
	@echo "   📂 Destino: $(TARGET_DIR)"
	@cp -r "$(SOURCE_DIR)" "$(TARGET_DIR)"
	@echo "✅ Sucesso! Workspace clonado perfeitamente."

start:
	@# 1. Validação de Argumentos
	@if [ -z "$(workspace)" ]; then \
		echo "⚠️  Uso incorreto."; \
		echo "   Execute: make start workspace=nome_do_workspace"; \
		exit 1; \
	fi

	@# 2. Definição dos caminhos completos
	$(eval TARGET_WORKSPACE := $(WORKSPACES_ROOT)/$(workspace))

	@# 3. Verificação: A pasta de workspace existe?
	@if [ ! -d "$(TARGET_WORKSPACE)" ]; then \
		echo "❌ ERRO: O workspace não foi encontrado:"; \
		echo "   -> $(TARGET_WORKSPACE)"; \
		echo "   Verifique se o nome está correto dentro da pasta workspaces."; \
		exit 1; \
	fi

	@# 4. Execução do Script
	@echo "🚀 Iniciando o SCutie para o workspace '$(workspace)'..."
	@eval "$$(conda shell.bash hook)" && \
	conda activate cutie-with-point-negative-point-tracking && \
	python3 interactive_demo.py --workspace workspaces/$(workspace)