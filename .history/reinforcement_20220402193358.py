from Snake import Snake, Game

def generateActions():
    actions = []
    actions = ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'DOWN', 'LEFT',  'UP', 'UP','UP','UP','UP']
    return actions

if __name__ == "__main__":
    agent = Snake()
    environment = Game(agent)
    for action in generateActions():
        is_over, score = environment.play_step(action)
        print(is_over)
        if is_over:
            print("Game Over")
            environment.game_over()
            break
    print("Score: ", score)