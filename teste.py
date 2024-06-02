

class Client:
    def __init__(self, name) -> None:
        self.name = name
    
    def __str__(self) -> str:
        return self.name
    
    
class Robot(Client):
    def __init__(self, name, robot) -> None:
        self.robot = robot
        super().__init__(name)
        
    @property
    def client(self):
        return super()
        
    def __str__(self) -> str:
        return super().__str__() + '-' + self.robot
    
class Test:
    def __init__(self, client, name) -> None:
        self.client = client
        self.name = name
        
    def __str__(self) -> str:
        return self.client.__str__() + '-' + self.name
        
robot = Robot('multirotor', 'hydrone')
print(robot)

teste = Test(robot.client, 'shadow')
print(teste)

a = [1,2,3]
b = [4,5]
print( a+b)