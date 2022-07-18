import socket, threading, time
import warnings
warnings.filterwarnings('ignore')

def dealClient(sock, addr):
    """第四步： 接受传来的数据，并发送给对方数据"""
    print("收到来自%s的信息..." % str(addr))
    sock.send(b'Hello,I am server!')
    while True:
        data = sock.recv(1024)
        time.sleep(1)
        if not data or data.decode('utf-8') == 'exit':
            break
        print("-->>%s！" % data.decode('utf-8'))
        sock.send(("Loop_Msg: %s!" % data.decode('utf-8')).encode('utf-8'))
    """第五步： 关闭Socket"""
    sock.close()
    print("已接受，拜拜")
    
def server():
    addr = ("192.168.137.203", 8885)

    tcpServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpServerSocket.bind(addr)
    tcpServerSocket.listen(10)
    
    """第一步，创建一个基于IPv4和TCP协议的Socket，并进行IP (127.0.0.1为本机IP)和端口绑定"""
    #s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #s.bind(('172.20.10.8', 8885)) # 端口号一致就行
    """第二步：监听连接"""
    #s.listen(5)
    print("我正在快马加鞭，莫方。。。")
    while True:
        """第三步：接受一个新连接"""
        sock, addr =  tcpServerSocket.accept()
        """创建新进程来处理TCP连接"""
        t = threading.Thread(target=dealClient, args=(sock, addr))
        t.start()
        
if __name__ == '__main__':
    server()



