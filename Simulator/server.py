import socket,sys,json,pdb
from Communicator import Communicator
import argparse

class Server:
	def __init__(self):
		"""	
			Constructor. Initializes the communicator_list to [] and the NETWORK_TIMER to 60 and INITIALTIMER to 60
		Args:
			None
		Returns:
			None
		"""
		self.communicator_list = []
		self.NETWORK_TIMER = 10000
		self.INITIALTIMER = 10000	
	
	def BuildServer(self,port_no,num_clients):
		"""Builds The server on the port_number port_no for num_clients
		Args:
			port_no: (int) The port number
			num_clients: (int) The number of clients who would join (>= 2 for all practical purposes)
		Returns: 
			None		
		"""
		s = socket.socket()
		s.settimeout(self.NETWORK_TIMER)
		host = "0.0.0.0"
		self.port = port_no		
		s.bind((host,port_no))
		s.listen(5)
		self.client_count = 0
		self.CLOSE_NETWORK = False						
		while self.client_count < num_clients and (not self.CLOSE_NETWORK):
			try:			
				c,addr = s.accept()
			except:				
				self.CLOSE_NETWORK = True			
			if(not self.CLOSE_NETWORK):
				self.client_count += 1
				self.communicator_list.append(Communicator())
				self.communicator_list[-1].setSocket(c,self.NETWORK_TIMER)
		s.close()

	
	def setNetworkTimer(self,Time_in_seconds):
		self.NETWORK_TIMER = Time_in_seconds
	
	def getNetworkTimer(self):
		return self.NETWORK_TIMER

	def RecvDataFromClient(self,client_id):		
		"""Receives Data from Client client_id
		Args: 
			client_id: The integer index of a client
		Returns: 
			data: Received on the socket to client_id, None in case of an Error
		"""
		data = None
		if(client_id < len(self.communicator_list)):					
			data = self.communicator_list[client_id].RecvDataOnSocket()
			if(data is None):
				print('ERROR : TIMEOUT ON CLIENT NETWORK' + str(client_id) + ' END')
				self.CloseClient(client_id)
		return data

	def SendData2Client(self,client_id,data):		
		"""Sends data to the Client client_id. In case data was None, sends the 
		   appropriate data (with ACTION='KILLPROC') and closes the socket
		Args:
			client_id : (int) client_id
			data : The json file to be sent, or None in case of an Error
		Returns:
			success_flag : True if send was successful
		"""
		success_flag = False
		if(data is None):
			data = {'meta': 'TIMEOUT ON CLIENT NETWORK', 'action':'KILLPROC','data':''}
		else:
			data = json.loads(data)

		if(client_id < len(self.communicator_list)):			
			success_flag = self.communicator_list[client_id].SendDataOnSocket(json.dumps(data))			
			if(not success_flag):
				print('ERROR : COULD NOT SEND DATA TO CLIENT ' + str(client_id))
				self.CloseClient(client_id)
			elif((data['action'] == 'KILLPROC') or (data['action'] == 'FINISH')):
				self.CloseClient(client_id)			
		return success_flag

	def CloseClient(self,client_id):
		"""Closes the client with client_id client_id
		Args:
			client_id : (int) index of client
		Returns:
			None
		"""		
		if(client_id < len(self.communicator_list)):
			self.communicator_list[client_id] = None
	
	def CloseAllClients(self):
		"""Closes all clients in the communicator_list and resets the communicator_list
		Args:
			None
		Returns:
			None
		"""
		for idx in range(len(self.communicator_list)):
			if(not self.communicator_list[idx] is None):
				self.CloseClient(idx)
		self.communicator_list: list[Communicator] = []
	
	def SendInitError2Clients(self):	
		"""
			In case of an initialization error, sends messages to the clients, and exits
		Args:
			None
		Returns: 
			None
		"""	
		for idx in range(len(self.communicator_list)):
			if(not self.communicator_list[idx] is None):
				data = {'meta':'ERROR IN INITIALIZATION', 'action':'KILLPROC','data':''}
				self.SendData2Client(idx,json.dumps(data))
				self.CloseClient(idx)

	def playTak(self,n,timelimit,client_0,client_1):
		"""
			"Tak is the best sort of game: simple in its rules, complex in its strategy" - Kvothe
			Starts a game of Tak between client_0 (as Player_1) and client_1 (as Player_2)
		Args:
			n: (int) board size
			timelimit: time limit 
			client_0: (int) idx of Player 1
			client_1: (int) idx of Player 2
		Returns:
			None
		"""
		if( (client_0 < len(self.communicator_list)) and (client_1) < len(self.communicator_list)):
			dataString = '1 ' + str(n) + ' ' + str(timelimit)
			data = {'meta':'', 'action':'INIT','data':dataString}
			self.SendData2Client(client_0, json.dumps(data))
			data['data'] = '2 ' + str(n) + ' ' + str(timelimit)
			self.SendData2Client(client_1, json.dumps(data))			
			while(True):
				data = self.RecvDataFromClient(client_0)
				self.SendData2Client(client_1, data)
				if not data:
					break
				print(data, 'Received from client 0')
				data = json.loads(data)
				if data['action'] == 'FINISH' or data['action'] == 'KILLPROC':
					break		
				data = self.RecvDataFromClient(client_1)
				print(data, 'Received from client 1')
				self.SendData2Client(client_0, data)
				if not data:
					break
				data = json.loads(data)
				if data['action'] == 'FINISH' or data['action'] == 'KILLPROC':
					break
			self.CloseClient(client_0)
			self.CloseClient(client_1)
		else:
			# Close all clients
			self.CloseAllClients()

if __name__ == '__main__':
	print('Start')
	local_Server = Server()
	parser = argparse.ArgumentParser(description = 'Tak Server')
	parser.add_argument('port', metavar = '10000', type = int, help = 'Server port')
	parser.add_argument('-n', dest = 'n', metavar = 'N', type = int, default = 5, help = 'Tak board size')
	parser.add_argument('-NC', dest = 'num_clients', metavar = 'num_clients', type = int, default = 2, help = 'Number of clients connecting to the server')
	parser.add_argument('-TL', dest = 'time_limit', metavar = 'time_limit', type = int, default = 120, help = 'Time limit (in s)')
	args = parser.parse_args()
	if args.n < 5 or args.n > 7:
		print('Game size should be 5x5, 6x6 or 7x7.')
		sys.exit()
	local_Server.BuildServer(args.port, args.num_clients)
	if(local_Server.client_count < 2):
		local_Server.SendInitError2Clients()
	else:
		local_Server.playTak(args.n,args.time_limit,0,1)	
