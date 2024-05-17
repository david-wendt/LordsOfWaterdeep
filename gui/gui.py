import tkinter as tk

class BoardGameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lords of Waterdeep")

        # Define board and player area dimensions
        self.unit = 30
        self.board_width = 36 * self.unit
        self.board_height = 16 * self.unit
        self.player_area_height = 200

        # Create canvases for the board and player area
        self.board_canvas = tk.Canvas(root, width=self.board_width, height=self.board_height, bg="tan")
        self.board_canvas.pack()

        self.player_canvas = tk.Canvas(root, width=self.board_width, height=self.player_area_height, bg="grey")
        self.player_canvas.pack()

        self.draw_vp_track(self.board_height, self.board_width, self.unit)
        self.draw_buildings()
        self.draw_player_mats()

    def draw_vp_track(self, height, width, tile_size):
        
        # Top border: 0 to 35
        for i in range(35):
            x = i * tile_size
            self.board_canvas.create_rectangle(x, 0, x + tile_size, tile_size, fill="white")
            self.board_canvas.create_text(x + tile_size // 2, tile_size // 2, text=str(i))
        
        # Right border: 36 to 50
        for i in range(35,51):
            y = (i - 35) * tile_size
            self.board_canvas.create_rectangle(width - tile_size, y, width, y + tile_size, fill="white")
            self.board_canvas.create_text(width - tile_size // 2, y + tile_size // 2, text=str(i))
        
        # Bottom border: 51 to 85
        for i in range(51,86):
            x = (width - tile_size) - (i - 51) * tile_size
            self.board_canvas.create_rectangle(x - tile_size, height - tile_size, x, height, fill="white")
            self.board_canvas.create_text(x - tile_size // 2, height - tile_size // 2, text=str(i))
        
        # Left border: 86 to 99
        for i in range(86,100):
            y = (height - tile_size) - (i - 85) * tile_size
            self.board_canvas.create_rectangle(0, y, tile_size, y + tile_size, fill="white")
            self.board_canvas.create_text(tile_size // 2, y + tile_size // 2, text=str(i))

    def draw_resource(self, x, y, color, width=20):
        self.board_canvas.create_rectangle(x,y, x+width, y+width, fill=color)

    def draw_buildings(self):

        #################################################
        # ---------- CLIFFWATCH INN (QUESTS) ---------- #
        #################################################

        self.board_canvas.create_rectangle(45, 45, 840, 240, outline="black", width=2)
        self.board_canvas.create_rectangle(60, 60, 240, 225, outline="black", width=2)
        self.board_canvas.create_rectangle(255, 60, 435, 225, outline="black", width=2)
        self.board_canvas.create_rectangle(450, 60, 630, 225, outline="black", width=2)
        self.board_canvas.create_rectangle(645, 60, 825, 225, outline="black", width=2)

        self.draw_resource(850, 55, 'gold')
        self.draw_resource(850, 85, 'gold')
        self.board_canvas.create_rectangle(880,65, 910, 95, outline='black')
        self.board_canvas.create_text(895, 80, text='Q', font=('Georgia', 25))
        self.draw_agent_space(920, 55)

        yshift = 60
        self.board_canvas.create_rectangle(845,65 + yshift, 875, 95 + yshift, outline='red')
        self.board_canvas.create_text(860, 80 + yshift, text='I', font=('Georgia', 25), fill='red')
        self.board_canvas.create_rectangle(880,65 + yshift, 910, 95 + yshift, outline='black')
        self.board_canvas.create_text(895, 80 + yshift, text='Q', font=('Georgia', 25))
        self.draw_agent_space(920, 55 + yshift)

        self.board_canvas.create_rectangle(845,65 + 2 * yshift, 875, 95 + 2 * yshift, outline='black')
        self.board_canvas.create_text(860, 80 + 2 * yshift, text='Reset\nquests', font=('Georgia', 10), fill='black')
        self.board_canvas.create_rectangle(880,65 + 2 * yshift, 910, 95 + 2 * yshift, outline='black')
        self.board_canvas.create_text(895, 80 + 2 * yshift, text='Q', font=('Georgia', 25))
        self.draw_agent_space(920, 55 + 2 * yshift)

        ################################################
        #### ----- DEFAULT RESOURCE BUILDINGS ----- ####
        ################################################

        # Field of Triumph
        self.draw_resource(45, 255, 'orange')
        self.draw_resource(75, 255, 'orange')
        self.draw_agent_space(45, 285)

        # Blackstaff Tower
        self.draw_resource(60, 355, 'purple')
        self.draw_agent_space(45, 385)

        # The Grinning Lion Tavern
        self.draw_resource(45 + 65, 255, 'black')
        self.draw_resource(75 + 65, 255, 'black')
        self.draw_agent_space(45 + 65, 285)

        # The Plinth
        self.draw_resource(60 + 65, 355, 'white')
        self.draw_agent_space(45 + 65, 385)

    def draw_player_mats(self, nplayers=4):
        pool_y_start = 20
        player_colors = ["blue", "red", "green", "black", "yellow"]
        player_height = 50

        for i, color in enumerate(player_colors):
            y = pool_y_start + i * player_height
            self.player_canvas.create_text(50, y, text=f"Player {i + 1}", fill=color, font=("Arial", 12))
            for j in range(5):  # 5 resource squares for each player
                self.player_canvas.create_rectangle(100 + j * 25, y - 10, 120 + j * 25, y + 10, fill="grey")
            for k in range(3):  # 3 round tokens for each player
                self.player_canvas.create_oval(250 + k * 30, y - 10, 270 + k * 30, y + 10, fill=color)
            self.player_canvas.create_text(400, y, text=f"Score: {i * 10}", fill="black", font=("Arial", 12))

    def draw_agent_space(self, x, y, inner_width=30, outer_width=50):
        inner_shift = (outer_width - inner_width) // 2
        self.board_canvas.create_rectangle(x, y, x + outer_width, y + outer_width, fill='gray')
        self.board_canvas.create_oval(x + inner_shift, y + inner_shift, 
                                      x + inner_shift + inner_width, y + inner_shift + inner_width, 
                                      fill='tan')

if __name__ == "__main__":
    root = tk.Tk()
    app = BoardGameGUI(root)
    root.mainloop()
