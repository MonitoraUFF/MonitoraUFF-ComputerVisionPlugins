# importe os pacotes necessários
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker():

	def __init__(self, maxDisappeared=15):
		# inicializa o próximo ID de objeto exclusivo junto com dois pedidos
		# dicionários usados ​​para rastrear o mapeamento de um determinado objeto
		# ID para seu centroide e número de quadros consecutivos que possui
		# foi marcado como "desaparecido", respectivamente
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		# armazena o número máximo de quadros consecutivos em um dado
		# objeto pode ser marcado como "desaparecido" até que
		# precisa cancelar o registro do objeto de rastreamento
		self.maxDisappeared = maxDisappeared

	def register(self, centroid):
		# ao registrar um objeto, usamos o próximo objeto disponível
		# ID para armazenar o centróide
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# para cancelar o registro de um ID de objeto, excluímos o ID do objeto de
		# ambos os nossos respectivos dicionários
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, centroids):
		# verifique se a lista de retângulos da caixa delimitadora de entrada
		# está vazia
		if len(centroids) == 0:
			# faz um loop sobre quaisquer objetos rastreados existentes e os marca
			# desapareceu
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				# se tivermos atingido um número máximo de consecutivas
				# frames onde um determinado objeto foi marcado como
				# ausente, cancele o registro
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
			# volte mais cedo, pois não há centróides ou informações de rastreamento
			# atualizar
			return self.objects
        # inicializa uma matriz de centróides de entrada para o quadro atual
		inputCentroids = np.zeros((len(centroids), 2), dtype="int")
		# loop sobre os retângulos da caixa delimitadora
		for (i, (center_x, center_y)) in enumerate(centroids):
			# use as coordenadas da caixa delimitadora para derivar o centróide
			inputCentroids[i] = (center_x, center_y)
        # se não estivermos rastreando nenhum objeto, pegue a entrada
		# centroids e registre cada um deles
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])
        # caso contrário, estamos rastreando objetos, então precisamos
		# tenta combinar os centróides de entrada com o objeto existente
		# centroids
		else:
			# pegue o conjunto de IDs de objeto e centróides correspondentes
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())
			# calcula a distância entre cada par de objetos
			# centróides e centróides de entrada, respectivamente - nosso
			# objetivo será combinar um centroide de entrada com um existente
			# centróide de objeto
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			# para realizar esta correspondência, devemos (1) encontrar o
			# menor valor em cada linha e (2) classificar a linha
			# índices com base em seus valores mínimos para que a linha
			# com o menor valor está na * frente * do índice
			# Lista
			rows = D.min(axis=1).argsort()
			# a seguir, realizamos um processo semelhante nas colunas por
			# encontrar o menor valor em cada coluna e depois
			# classificação usando a lista de índice de linha computada anteriormente
			cols = D.argmin(axis=1)[rows]
            # para determinar se precisamos atualizar, registrar,
			# ou cancele o registro de um objeto que precisamos para manter o controle de qual
			# das linhas e índices de coluna que já examinamos
			usedRows = set()
			usedCols = set()
			# faz um loop sobre a combinação do índice (linha, coluna)
			# tuplas
			for (row, col) in zip(rows, cols):
				# se já examinamos a linha ou
				# valor da coluna antes, ignore-o
				# val
				if row in usedRows or col in usedCols:
					continue
				# caso contrário, pegue o ID do objeto para a linha atual,
				# definir seu novo centróide e redefinir o desaparecido
				# contador
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0
				# indica que examinamos cada uma das linhas e
				# índices de coluna, respectivamente
				usedRows.add(row)
				usedCols.add(col)
            # calcula o índice de linha e coluna que ainda NÃO temos
			# examinado
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            # no caso de o número de centróides do objeto ser
			# igual ou maior que o número de centróides de entrada
			# precisamos verificar e ver se alguns desses objetos têm
			# desapareceu potencialmente
			if D.shape[0] >= D.shape[1]:
				# faz um loop sobre os índices de linha não utilizados
				for row in unusedRows:
					# pegue o ID do objeto para a linha correspondente
					# indexar e incrementar o contador desaparecido
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					# verifique se o número de
					# frames o objeto foi marcado como "desaparecido"
					# para autorizações de cancelamento de registro do objeto
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
            # caso contrário, se o número de centróides de entrada for maior
			# do que o número de centróides de objetos existentes que precisamos
			# registra cada novo centroide de entrada como um objeto rastreável
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])
		# retorna o conjunto de objetos rastreáveis
		return self.objects
