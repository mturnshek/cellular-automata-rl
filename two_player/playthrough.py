def single_playthrough(red_agent, blue_agent, env):
    max_moves = 1000
    move_count = 0
    while not env.is_done():
        if env.game.turn == 'red':
            cell = red_agent.act(env.state())
        else:
            cell = blue_agent.act(env.state())
        env.step(cell)
        move_count += 1
        if move_count > max_moves:
            break

    if env.reward('red') == 1:
        return red_agent
    else:
        return blue_agent
