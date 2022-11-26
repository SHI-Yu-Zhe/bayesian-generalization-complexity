using Images

# This is only for verifying the correctness of our python entropy calculator.

println(readdir("./imgs/"))

for i in readdir("./imgs/")
    img = load("./imgs/$i")
    # extract r, g, b channels
    img_channels = channelview(img)

    ent = (entropy(img_channels[1, :, :]) + entropy(img_channels[1, :, :]) + entropy(img_channels[1, :, :])) / 3
    println("Ent $i: $ent")
end
