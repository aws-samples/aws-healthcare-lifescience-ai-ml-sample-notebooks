import gradio as gr
from fastapi import FastAPI
from PIL import Image
import time

from anthropic import AnthropicBedrock
anthropic_client = AnthropicBedrock()

app = FastAPI()

def get_llm_response(prompt_data):
    for iteration in range(10):
        try:
            message = anthropic_client.messages.create(
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                max_tokens=2000,
                temperature=0,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt_data}]}
                ],
            )
            # Concatenate the text from all items in the message content
            message_content = ''.join(item.text for item in message.content if hasattr(item, 'text'))
            return message_content  # Return the concatenated message content
        except Exception as ex:
            if "ThrottlingException" in str(ex):
                print(f"Throttled! Sleeping attempt #{iteration + 1}...")
                time.sleep(5)  # Wait for 5 seconds before retrying
            else:
                print(f"Encountered an exception: {ex}")
                raise ex# Exit the loop on non-throttling exceptions

    return ""  # Return an empty string if unable to get a response


def get_llm_results(Manufacturing_Protocol, Standard_Operating_Procedure):
    try:
        prompt_data = f"""I want you to first read the following two passages. \nThe first passage, called "Procedure", contains manufacturing instructions for a pharmaceutical product. \nThe second passage, called "Rules", contains a list of rules that **regulate** manufacturing of the a pharmaceutical product. \nAfter you read the Procedure and Rules passages, I want you to write a report that contains all the times that the Procedure violate the Rules. Please include where specifically the Procedure and Rules are in conflict. I want you to think carefully about this, noting even subtle contradictions. \nPlease note that there many not be any violations. In which case I want you to simply return "no violations found" \n\nProcedure: \n{Manufacturing_Protocol} \n\nRules: \n{Standard_Operating_Procedure} \n\nAssistant:"""
        response = get_llm_response(prompt_data)
    except Exception as ex:
        print(f"Encountered an exception: {ex}")
        raise ex
    return response


def get_llm_results_amelioration(Manufacturing_Protocol, Potential_Errors):
    try:
        prompt_data = f"I want you to first read the following two passages. \nThe first passage, called 'Procedure', contains manufacturing instructions for a pharmaceutical product. \nThe second passage, called 'Potential Violations', contains a list of potential violations of certain rules that regulate the procedures. \n\nAfter you read the Procedure and Potential Violations passages, I want you to re-write the Procedure, \n but make it so that the Procedure is now consistent with any Potential Violations. I want you to keep the Procedure as close as possible to the original, \n`but fix any violations if they exist. I want you to put any changes you make in '**'. For example, if you change 50 to 100, it should be '**100**'.\n\nPlease note that there may not be any violations. In which case I want you to simply return the procedure as is.  \n\nProcedure:\n\n{Manufacturing_Protocol}\n\nRules:\n\n{Potential_Errors}\n\nAssistant:\n\n"
        response = get_llm_response(prompt_data)
    except Exception as ex:
        print(f"Encountered an exception: {ex}")
        raise ex
    return response


def update_textbox(dropdown_selection):
    mp_1 = """
    Penicillin Manufacturing Protocal 

    Materials: 
    - Penicillium chrysogenum mold strain (ATCC 48271 or equivalent)
    - Growth medium:
    -- Corn steep liquor (5-10% w/v) 
    -- Sucrose (2-5% w/v)
    -- Ammonium sulfate (0.5-1% w/v) 
    -- Potassium phosphate (0.1-0.5% w/v)
    -- Fermentation vessel (100-500L capacity)
    -- Centrifuge
    -- Rotary evaporator 
    -- Ion exchange resin (strongly acidic cation exchange resin)
    -- Activated charcoal 
    -- Reverse osmosis system
    -- Sterile 0.9% sodium chloride solution

    Method:
    1. Inoculate a slant or plate of P. chrysogenum and incubate at 25°C for 3-5 days until sporulation occurs.
    2. Inoculate a starter culture of the growth medium with P. chrysogenum spores and incubate at 25°C for 2 days on a rotary shaker (200rpm) until a cell density of 1-5 x 107 CFU/mL is reached. 
    3. Inoculate the fermentation vessel with 10% v/v of the starter culture. 
    4. Incubate the fermentation vessel at 25°C for 5-7 days while aerating (1 vvm) and stirring (200rpm) until maximum penicillin titre is reached (100-500 IU/mL). 
    5. Centrifuge the fermentation broth at 10000xg for 20 minutes to remove cells and debris.
    6. Concentrate the supernatant using a rotary evaporator to remove excess water.
    7. Pass the concentrate through an ion exchange resin to remove impurities.
    8. Pass the concentrate through activated charcoal to remove pigments and odorous compounds.
    9. Concentrate and wash the product using a reverse osmosis system. 
    10. Re-suspend the product in sterile 0.9% sodium chloride solution to achieve a concentration of 100,000 IU penicillin G per mL.
    11. Filter sterilize the product through a 0.22μm membrane and store at 2-8°C.
    """

    sop_1 = """
    1. All incubations must be less than 2 days.  
    2. All sodium chloride must be greater than .95% solutions
    3. No batch can exceed 500 liters in volume. 
    4. All filtration must use 0.2 micron filters or smaller.
    5. No raw material can be used after 6 months from receipt. 
    6. All equipment must be sterilized at 121°C for at least 15 minutes.
    7. No more than 2 different products can be manufactured in the same facility. 
    8. All surfaces must be wiped down with 70% isopropyl alcohol. 
    9. No batch record can have more than 10 deviations noted.
    10. All finished products must have at least 2 years of shelf life remaining at time of release.

    """

    mp_2 = """
    1. Cell culture and expansion - The hybridoma cell line expressing the anti-HER2 antibody is expanded in cell culture using Dulbecco's Modified Eagle's Medium (DMEM) supplemented with 10% fetal bovine serum (FBS) and 1% penicillin-streptomycin. Cells are grown in T-flasks in a humidified 37°C incubator with 5% CO2. Once 70-80% confluent, cells are passaged to maintain cell density between 1-3 million cells/mL.  
2. Bioreactor culture - Once sufficient cells have been expanded, they are transferred to a bioreactor for large-scale production of the antibody. Cells are cultured in a fed-batch bioreactor using a proprietary chemically defined culture medium. Critical parameters like pH, dissolved oxygen, temperature, and agitation are closely monitored and controlled. Growth factors and nutrients are added to sustain high cell viability and maximize antibody production.
3. Clarification - Once maximum cell density and antibody titer have been achieved, the bioreactor harvest is clarified using depth filtration to remove cells and cellular debris. The clarified harvest is then concentrated using tangential flow filtration. 
4. Purification - The concentrated antibody solution is purified using affinity chromatography, ion exchange chromatography and size exclusion chromatography. The affinity resin specifically binds the Fc region of antibodies, separating them from other proteins. Ion exchange separates antibody charge variants. Size exclusion further purifies the antibody and also acts as a viral clearance step. 
5. Formulation and fill - The purified bulk antibody is buffer exchanged into a stabilizing formulation buffer and sterile filtered. It is then aseptically filled into sterile glass vials and stored for distribution.
6. Quality control - Rigorous quality control testing is performed at multiple stages of the process including purity, potency, identity, safety, and stability. Only product that meets pre-determined specifications is released for clinical or commercial use.
"""
    mp_3 = """
    Manufacturing Process for Insulin Pump Model XR7 

1. Prepare polymer solution for pump casing
- Mix 300g of polycarbonate resin with 400mL of dichloromethane solvent 
- Agitate solution at 600rpm for 30 minutes to ensure homogeneity
- Degas solution under vacuum for 15 minutes to remove air bubbles

2. Injection mold pump casing and components 
- Load polymer solution into injection molding machine hopper
- Heat solution to 425°C to reach proper viscosity for molding 
- Inject solution into mold cavities for pump casing, control interface, and tubing ports 
- Once cooled, eject molded parts from mold. Inspect for defects and set aside

3. Assemble electronic components
- Solder circuit boards for control interface, processor, and batteries 
- Attach LCD display to control interface board 
- Connect all boards using ribbon cables and test for continuity and function
- Seal boards in epoxy resin to waterproof components

4. Attach molded parts and tubing  
- Connect control interface to pump casing 
- Attach tubing ports to pump casing 
- Cut PVC tubing to required lengths and connect between tubing ports
- Attach batteries to control interface and seal pump casing

5. Calibrate and test device
- Connect insulin reservoir and load pump with test fluid
- Set control interface to deliver 1.5U/hour basal rate
- Run pump for 30 minutes and verify proper fluid delivery rate 
- Make any final adjustments to device settings or tubing length
- Package final product for shipping and inspect one more time for quality 
"""
    sop_2 = """
    
Regulatory Specifications for Insulin Pump Manufacturing 

1. Materials 
1.1 All materials used in the construction of the insulin pump must be biocompatible per ISO 10993 standards and not leach any toxic, carcinogenic or biologically reactive compounds when in contact with interstitial insulin and tissue fluids. 
1.2 The housing and durable components shall be made of medical-grade stainless steel or titanium alloys to withstand a minimum of 3 years of continuous use.
1.3 All tubing, gaskets, seals and disposable components must be made of medical-grade low-density polyethylene, polypropylene or other polymers that are compatible with insulin solutions.
1.4 Pump must be tested for a total of 1 hour to ensure proper functioning.
1.5 All electronic components must meet IPC-A-610 Class 2 standards for electronic assemblies.


"""
    if dropdown_selection == "Penicillin Synthesis Example":
        return [mp_1, sop_1]
    elif dropdown_selection == "Monoclonal Antibody Culture Example":
        return [mp_2, sop_1]
    elif dropdown_selection == "Insulin Pump Manufacturing Example":
        return [mp_3, sop_2]

theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="purple",
    neutral_hue="slate",
)

with gr.Blocks(theme=theme) as ui:

    mp_1 = """
    Manufacturing Process for Insulin Pump Model XR7 

1. Prepare polymer solution for pump casing
- Mix 300g of polycarbonate resin with 400mL of dichloromethane solvent 
- Agitate solution at 600rpm for 30 minutes to ensure homogeneity
- Degas solution under vacuum for 15 minutes to remove air bubbles

2. Injection mold pump casing and components 
- Load polymer solution into injection molding machine hopper
- Heat solution to 425°C to reach proper viscosity for molding 
- Inject solution into mold cavities for pump casing, control interface, and tubing ports 
- Once cooled, eject molded parts from mold. Inspect for defects and set aside

3. Assemble electronic components
- Solder circuit boards for control interface, processor, and batteries 
- Attach LCD display to control interface board 
- Connect all boards using ribbon cables and test for continuity and function
- Seal boards in epoxy resin to waterproof components

4. Attach molded parts and tubing  
- Connect control interface to pump casing 
- Attach tubing ports to pump casing 
- Cut PVC tubing to required lengths and connect between tubing ports
- Attach batteries to control interface and seal pump casing

5. Calibrate and test device
- Connect insulin reservoir and load pump with test fluid
- Set control interface to deliver 1.5U/hour basal rate
- Run pump for 30 minutes and verify proper fluid delivery rate 
- Make any final adjustments to device settings or tubing length
- Package final product for shipping and inspect one more time for quality 
    """

    sop_1 = """
Regulatory Specifications for Insulin Pump Manufacturing 

1. Materials 
1.1 All materials used in the construction of the insulin pump must be biocompatible per ISO 10993 standards and not leach any toxic, carcinogenic or biologically reactive compounds when in contact with interstitial insulin and tissue fluids. 
1.2 The housing and durable components shall be made of medical-grade stainless steel or titanium alloys to withstand a minimum of 3 years of continuous use.
1.3 All tubing, gaskets, seals and disposable components must be made of medical-grade low-density polyethylene, polypropylene or other polymers that are compatible with insulin solutions.
1.4 Pump must be tested for a total of 1 hour to ensure proper functioning.
1.5 All electronic components must meet IPC-A-610 Class 2 standards for electronic assemblies.

    """

    header = gr.Markdown(
        """
    # Automated Checking of Pharmaceutical Manufacturing Compliance
    
    Pharmaceutical Manufacturing faces a complex regulatory and quality hurdle since there is a pressing need to ensure that manufacturing processes adhere to regulatory frameworks. Frequently, manual analysis of Standard Operating Procedures (SOPs) that govern Manufacturing Protocols is required. 
    
    This demo shows how a Generative AI approach can be used to instead automatically check protocols for contradictions or violations of standard operation procedures. This can be schematically represented as:
    """
    )
    # img = Image.open('/deployment/images/manufacturing_diagram.png')
    img = Image.open("images/manufacturing_diagram.png")  # for dev
    img.align = "center"
    image = gr.Image(img, height=278, width=976)

    header = gr.Markdown(
        """
    Enter a Manufacturing Protocol in the first box and a Standard Operating Procedure (SOP) in the second box. The manufacturing protocol can be a list of instructions or procedures. The Standard Operating Procedure should be a list of specifications that regulate the manufacturing procedure. The Output will be the components of the Manufacturing Protocal that are predicted to violate the Standard Operating Procedure. This demo uses Amazon Bedrock (Claude) to then check for SOP Violations.
    
    You can either enter your own text, or select a few different examples from the bottom of this page.
    
    """
    )
    mp = gr.Textbox(label="Manufacturing Protocol", value=mp_1)
    sop = gr.Textbox(label="Standard Operation Procedure", value=sop_1)
    options = [
        "Penicillin Synthesis Example",
        "Monoclonal Antibody Culture Example",
        "Insulin Pump Manufacturing Example",
    ]
    dropdown = gr.Dropdown(options, label="Options")
    update_button = gr.Button("Update Examples", variant="secondary")
    update_button.click(update_textbox, inputs=dropdown, outputs=[mp, sop])

    output = gr.Textbox(label="Predicted Standard Operating Procedures Report")
    greet_btn = gr.Button("Find SOP Violations", variant="primary")
    greet_btn.click(
        fn=get_llm_results,
        inputs=[mp, sop],
        outputs=output,
        api_name="find_violations",
        scroll_to_output=True,
    )

    output2 = gr.Textbox(label="Ameliorated Protocol")
    ameliorate_btn = gr.Button("Fix Manufacturing Protocol", variant="primary")
    ameliorate_btn.click(
        fn=get_llm_results_amelioration,
        inputs=[mp, output],
        outputs=output2,
        api_name="fix_violation",
        scroll_to_output=True,
    )

# serve the app
if __name__ == "__main__":
    ui.launch(
        share=False,
        server_name="0.0.0.0",
        auth=("admin", "password"),
        server_port=8080,
    )
