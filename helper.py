def amplitude_function(axs):
    FONT_SIZE = 12
    TEXT_COLOR = "black"
    LABEL_SIZE = 10
    axs.set_title("Amplitude", fontsize=FONT_SIZE, fontweight="bold", color=TEXT_COLOR)
    axs.set_xlabel("X (Traço)", fontsize=FONT_SIZE, color=TEXT_COLOR)
    axs.set_ylabel("Profundidade (m)", fontsize=FONT_SIZE, color=TEXT_COLOR)
    axs.set_yticklabels(
        [str(int(tick) * 4) for tick in axs.get_yticks()]
    )  # Altera os rótulos do eixo y
    axs.tick_params(axis="both", labelsize=LABEL_SIZE)
