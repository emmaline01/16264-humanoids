import random

class Cactus():
    def __init__(self, canvas, groundY, img, dinoCanvas, dinoImg):
        self.dx = -5
        self.img = img
        self.dinoImg = dinoImg
        self.canvasObj = canvas.create_image(220, groundY, image=img, anchor="s")
        self.startX, _, *_ = canvas.bbox(self.canvasObj)
        self.x = self.startX
        self.onScreen = False
        self.canvas = canvas
        self.dinoCanvas = dinoCanvas

        self.gameOver = False

    def setGameOver(self, gameOver):
        self.gameOver = gameOver

    def getGameOver(self):
        return self.gameOver

    def checkDinoCollision(self):
        

        # lx, ly, rx, ry = self.canvas.bbox(self.canvasObj)
        # dinoLX, dinoLY, dinoRX, dinoRY = self.canvas.bbox(self.dinoCanvas)
        # self.canvas.create_rectangle(lx, ly, rx, ry, outline="#fb0", fill="#fb0")
        # self.canvas.create_rectangle(dinoLX, dinoLY, dinoRX, dinoRY, outline="#05f", fill="#05f")
        
        # L is bottom left point, R is top right point
        
        c = self.canvas.coords(self.canvasObj)
        lx, ly, rx, ry = c[0] - self.img.width()/2, c[1], self.img.width()/2 + c[0], c[1] - self.img.height()
        # self.canvas.create_rectangle(lx, ly, rx, ry, outline="#fb0", fill="#fb0")

        c2 = self.canvas.coords(self.dinoCanvas)
        dinoLX, dinoLY, dinoRX, dinoRY = c2[0] - self.dinoImg.width()/2, c2[1], c2[0] + self.dinoImg.width()/2, c2[1] - self.dinoImg.height()
        # self.canvas.create_rectangle(dinoLX, dinoLY, dinoRX, dinoRY, outline="#05f", fill="#05f")

        # print(ly, ry)
        # print("dino", dinoLY, dinoRY)
        # print(self.canvas.coords(self.canvasObj), self.img.width(), self.img.height())
        # print(self.canvas.coords(self.dinoCanvas), self.dinoImg.width(), self.dinoImg.height())
        
        # check x bounds
        if ((lx <= dinoRX and lx >= dinoLX) or 
            (rx <= dinoRX and rx >= dinoLX)):
            # check y bounds
            if ((ly <= dinoLY and ly >= dinoRY) or 
                (ry <= dinoLY and ry >= dinoRY)):
                return True
        return False


    # update by scrolling across screen or starting to scroll across screen
    def update(self):
        if self.onScreen:
            self.canvas.move(self.canvasObj, self.dx, 0)
            self.x += self.dx
            if self.checkDinoCollision():
                self.gameOver = True

            # check if offscreen
            if (self.x < -20):
                self.onScreen = False
                self.canvas.move(self.canvasObj, self.startX - self.x, 0)
                self.x = self.startX
        else:
            if (random.random() < 0.2): #0.2 probability of spawning
                self.onScreen = True